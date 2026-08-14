import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import shutil
import tempfile
import random
import numpy as np
import torch
import torch.nn as nn
import model_compression_toolkit as mct
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.target_platform_capabilities.constants import TFLITE_TP_MODEL
from model_compression_toolkit.target_platform_capabilities.tpc_models.tflite_tpc.v1_0.tpc import get_op_quantization_configs
import onnx2tf
import tensorflow as tf

from classes import CLASS_NAMES
from keras_pipeline.export_validator import inspect_tflite, write_tflite_sidecar_manifest
from pytorch_pipeline.model import get_anti_spoof_model

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def get_npu_tflite_tpc(name="npu_tflite_tpc"):
    """
    Sony MCT TFLite TargetPlatformCapabilities를 생성합니다.
    (quantization_preserving 설정 시 enable_activation_quantization=False 양립 버그 수정 포함)
    """
    linear_eight_bits, mixed_precision_cfg_list, eight_bits_default = get_op_quantization_configs()

    default_configuration_options = schema.QuantizationConfigOptions(
        quantization_configurations=tuple([eight_bits_default])
    )
    base_configuration_options = schema.QuantizationConfigOptions(
        quantization_configurations=tuple([linear_eight_bits]),
        base_config=linear_eight_bits
    )

    operator_set = []
    fusing_patterns = []

    # quantization_preserving 활성화 시 enable_activation_quantization은 반드시 False여야 함
    quant_preserving = default_configuration_options.clone_and_edit(
        quantization_preserving=True,
        enable_activation_quantization=False
    )

    for op_name in [
        schema.OperatorSetNames.UNSTACK, schema.OperatorSetNames.TRANSPOSE,
        schema.OperatorSetNames.GATHER, schema.OperatorSetNames.RESHAPE,
        schema.OperatorSetNames.MAXPOOL, schema.OperatorSetNames.AVGPOOL,
        schema.OperatorSetNames.STRIDED_SLICE, schema.OperatorSetNames.CONCATENATE,
        schema.OperatorSetNames.MUL, schema.OperatorSetNames.MIN,
        schema.OperatorSetNames.MAX, schema.OperatorSetNames.ZERO_PADDING2D,
        schema.OperatorSetNames.RESIZE, schema.OperatorSetNames.PAD,
        schema.OperatorSetNames.FOLD
    ]:
        operator_set.append(schema.OperatorsSet(name=op_name, qc_options=quant_preserving))

    operator_set.append(schema.OperatorsSet(
        name=schema.OperatorSetNames.L2NORM,
        qc_options=default_configuration_options.clone_and_edit(
            fixed_zero_point=0, fixed_scale=1 / 128
        )
    ))
    operator_set.append(schema.OperatorsSet(
        name=schema.OperatorSetNames.LOG_SOFTMAX,
        qc_options=default_configuration_options.clone_and_edit(
            fixed_zero_point=127, fixed_scale=16 / 256
        )
    ))
    operator_set.append(schema.OperatorsSet(
        name=schema.OperatorSetNames.SOFTMAX,
        qc_options=default_configuration_options.clone_and_edit(
            fixed_zero_point=-128, fixed_scale=1 / 256
        )
    ))

    sigmoid = schema.OperatorsSet(
        name=schema.OperatorSetNames.SIGMOID,
        qc_options=default_configuration_options.clone_and_edit_weight_attribute(
            weights_per_channel_threshold=False
        )
    )
    tanh = schema.OperatorsSet(
        name=schema.OperatorSetNames.TANH,
        qc_options=default_configuration_options.clone_and_edit(
            fixed_zero_point=-128, fixed_scale=1 / 256
        )
    )
    fc = schema.OperatorsSet(
        name=schema.OperatorSetNames.FULLY_CONNECTED,
        qc_options=base_configuration_options.clone_and_edit_weight_attribute(
            weights_per_channel_threshold=False
        )
    )
    squeeze = schema.OperatorsSet(
        name=schema.OperatorSetNames.SQUEEZE,
        qc_options=quant_preserving
    )

    conv2d = schema.OperatorsSet(name=schema.OperatorSetNames.CONV, qc_options=base_configuration_options)
    relu = schema.OperatorsSet(name=schema.OperatorSetNames.RELU)
    relu6 = schema.OperatorsSet(name=schema.OperatorSetNames.RELU6)
    elu = schema.OperatorsSet(name=schema.OperatorSetNames.ELU)
    batch_norm = schema.OperatorsSet(name=schema.OperatorSetNames.BATCH_NORM)
    add = schema.OperatorsSet(name=schema.OperatorSetNames.ADD)
    bias_add = schema.OperatorsSet(name=schema.OperatorSetNames.ADD_BIAS)

    kernel = schema.OperatorSetGroup(operators_set=[conv2d, fc])
    activations_to_fuse = schema.OperatorSetGroup(operators_set=[relu, elu])

    operator_set.extend([fc, conv2d, relu, relu6, tanh, sigmoid, batch_norm, add, bias_add, elu, squeeze])

    fusing_patterns.append(schema.Fusing(operator_groups=(kernel, bias_add)))
    fusing_patterns.append(schema.Fusing(operator_groups=(kernel, bias_add, activations_to_fuse)))
    fusing_patterns.append(schema.Fusing(operator_groups=(conv2d, batch_norm, activations_to_fuse)))
    fusing_patterns.append(schema.Fusing(operator_groups=(conv2d, squeeze, activations_to_fuse)))
    fusing_patterns.append(schema.Fusing(operator_groups=(batch_norm, activations_to_fuse)))
    fusing_patterns.append(schema.Fusing(operator_groups=(batch_norm, add, activations_to_fuse)))

    return schema.TargetPlatformCapabilities(
        default_qco=default_configuration_options,
        tpc_minor_version=1,
        tpc_patch_version=0,
        operator_set=tuple(operator_set),
        fusing_patterns=tuple(fusing_patterns),
        tpc_platform_type=TFLITE_TP_MODEL,
        add_metadata=False,
        name=name
    )


def _build_representative_dataset(dataset_dir="dataset/raw/train", model_type="crop_ir", num_samples=200):
    """
    Sony MCT 및 TFLite 양자화 보정용 대표 데이터셋 제너레이터를 구성합니다.
    실제 train 고정 split 이미지에서 stratified/sampled 프레임을 추출합니다.
    """
    from pytorch_pipeline.dataset import DualInputDataset, _get_default_transforms
    from utils import collect_split_items

    train_items = []
    # 1. dataset_dir이 split 폴더(예: dataset/raw/train)인 경우
    if os.path.isdir(os.path.join(dataset_dir, "live")):
        raw_root = os.path.dirname(dataset_dir)
        split_name = os.path.basename(dataset_dir)
        try:
            train_items = collect_split_items(raw_root, split_name)
        except Exception as e:
            print(f"[경고] collect_split_items({raw_root}, {split_name}) 실패: {e}")
    # 2. dataset_dir이 root 폴더(예: dataset/raw)인 경우
    elif os.path.isdir(os.path.join(dataset_dir, "train", "live")):
        try:
            train_items = collect_split_items(dataset_dir, "train")
        except Exception as e:
            print(f"[경고] collect_split_items({dataset_dir}, train) 실패: {e}")

    if train_items:
        train_transform_rgb, _, transform_ir = _get_default_transforms()
        dataset = DualInputDataset(
            train_items,
            transform_rgb=train_transform_rgb,
            transform_ir=transform_ir,
            augment=False
        )
        random.seed(42)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[:min(num_samples, len(dataset))]
        subset = torch.utils.data.Subset(dataset, indices)
        train_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
        print(f"[캘리브레이션 데이터셋] 실제 학습 데이터 {len(subset)}장 로드 완료 (모드: {model_type})")
    else:
        print("[경고] 실제 데이터셋을 찾을 수 없어 더미 데이터로 대체합니다.")
        train_loader = None

    def _gen():
        count = 0
        if train_loader is not None:
            for rgb_tensor, ir_tensor, _ in train_loader:
                if model_type in ("crop_ir", "single_ir"):
                    yield [ir_tensor]  # [1, 1, 224, 224]
                elif model_type in ("crop_rgb", "single_rgb"):
                    yield [rgb_tensor]  # [1, 3, 224, 224]
                elif model_type in ("dual", "dual_input"):
                    yield [rgb_tensor, ir_tensor]
                count += 1
                if count >= num_samples:
                    break
        else:
            for _ in range(num_samples):
                if model_type in ("crop_ir", "single_ir"):
                    yield [torch.randn(1, 1, 224, 224)]
                elif model_type in ("crop_rgb", "single_rgb"):
                    yield [torch.randn(1, 3, 224, 224)]
                elif model_type in ("dual", "dual_input"):
                    yield [torch.randn(1, 3, 224, 224), torch.randn(1, 1, 224, 224)]

    return _gen


def convert_pytorch_to_tflite(
    pth_path="model/pytorch/best_crop_ir_mobilenetv3_fixed.pth",
    output_prefix="model/pytorch/best_crop_ir_mobilenetv3_fixed",
    model_type="crop_ir",
    num_classes=len(CLASS_NAMES),
    calib_samples=200,
    dataset_dir="dataset/raw/train"
):
    """
    PyTorch 체크포인트를 Sony MCT를 거쳐 Float32, Full INT8, NPU INT8 TFLite 모델 및 사이드카 매니페스트로 변환합니다.
    """
    print(f"\n==================================================")
    print(f"[PyTorch → Sony MCT → TFLite 변환 시작]")
    print(f" - 체크포인트: {pth_path}")
    print(f" - 모델 타입: {model_type}")
    print(f" - 분류 클래스 수: {num_classes}")
    print(f" - 출력 접두사: {output_prefix}")
    print(f"==================================================")

    model = get_anti_spoof_model(model_type=model_type, num_classes=num_classes)
    if os.path.exists(pth_path):
        state_dict = torch.load(pth_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f" -> PyTorch 체크포인트 로드 완료: {pth_path}")
    else:
        print(f" -> [알림] {pth_path} 가 존재하지 않아 초기화된 가중치로 변환을 진행합니다.")
    model.eval()

    # 1. Representative Dataset 준비 (실제 얼굴 이미지 기반 캘리브레이션)
    rep_gen = _build_representative_dataset(
        dataset_dir=dataset_dir,
        model_type=model_type,
        num_samples=calib_samples
    )

    # 2. Sony MCT PTQ 수행
    print("\n[단계 1] Sony MCT PTQ (Post-Training Quantization) 실행 중...")
    tpc = get_npu_tflite_tpc()
    quantized_model, user_info = mct.ptq.pytorch_post_training_quantization(
        in_module=model,
        representative_data_gen=rep_gen,
        target_platform_capabilities=tpc
    )
    print(" -> Sony MCT PTQ 완료!")

    # 3. ONNX 내보내기 (PyTorch 2.x classic exporter 호환 및 텐서 계약 이름 적용)
    temp_dir = tempfile.mkdtemp(prefix="mct_pytorch_export_")
    onnx_path = os.path.join(temp_dir, "model_mct.onnx")

    _orig_onnx_export = torch.onnx.export
    def _compat_onnx_export(m, args, f, *opt_args, **opt_kwargs):
        if "dynamo" not in opt_kwargs:
            opt_kwargs["dynamo"] = False
        if model_type in ("crop_ir", "single_ir"):
            opt_kwargs["input_names"] = ["b_crop_ir"]
        elif model_type in ("crop_rgb", "single_rgb"):
            opt_kwargs["input_names"] = ["a_crop_rgb"]
        elif model_type in ("dual", "dual_input"):
            opt_kwargs["input_names"] = ["a_rgb", "b_ir"]
        opt_kwargs["output_names"] = ["logits"]
        return _orig_onnx_export(m, args, f, *opt_args, **opt_kwargs)
    torch.onnx.export = _compat_onnx_export

    print(f"\n[단계 2] ONNX 모델 내보내기 중 ({onnx_path})...")
    mct.exporter.pytorch_export_model(
        model=quantized_model,
        save_model_path=onnx_path,
        repr_dataset=rep_gen,
        serialization_format=mct.exporter.PytorchExportSerializationFormat.ONNX,
        quantization_format=mct.exporter.QuantizationFormat.FAKELY_QUANT,
        onnx_opset_version=17
    )
    print(f" -> ONNX 내보내기 완료 (크기: {os.path.getsize(onnx_path)} bytes)")

    # 4. onnx2tf를 통한 TensorFlow SavedModel 변환
    print("\n[단계 3] onnx2tf를 통한 SavedModel 변환 중...")
    saved_model_dir = os.path.join(temp_dir, "saved_model")
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=saved_model_dir,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True
    )

    out_dir = os.path.dirname(output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 5. TFLite 산출물 3종 생성 (Float32, INT8, NPU INT8)
    # 5-1. Float32 TFLite
    float_tflite_path = f"{output_prefix}_float.tflite"
    print(f"\n[단계 4-1] Float32 TFLite 생성: {float_tflite_path}")
    converter_float = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    float_model_bytes = converter_float.convert()
    with open(float_tflite_path, "wb") as f:
        f.write(float_model_bytes)
    inspect_tflite(float_tflite_path, model_type)
    write_tflite_sidecar_manifest(float_tflite_path, model_type)

    # 5-2. INT8 TFLite 캘리브레이션 제너레이터 (NHWC float32)
    def _tf_rep_dataset_gen():
        for sample in rep_gen():
            nhwc_sample = []
            for t in sample:
                arr = t.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
                nhwc_sample.append(arr)
            yield nhwc_sample

    # 5-3. Full INT8 TFLite
    int8_tflite_path = f"{output_prefix}_int8.tflite"
    print(f"\n[단계 4-2] Full INT8 TFLite 생성: {int8_tflite_path}")
    converter_int8 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = _tf_rep_dataset_gen
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.int8
    converter_int8.inference_output_type = tf.int8
    int8_model_bytes = converter_int8.convert()
    with open(int8_tflite_path, "wb") as f:
        f.write(int8_model_bytes)
    inspect_tflite(int8_tflite_path, model_type)
    write_tflite_sidecar_manifest(int8_tflite_path, model_type)

    # 5-4. NPU-friendly INT8 TFLite
    npu_int8_tflite_path = f"{output_prefix}_npu_int8.tflite"
    print(f"\n[단계 4-3] NPU INT8 TFLite 생성: {npu_int8_tflite_path}")
    with open(npu_int8_tflite_path, "wb") as f:
        f.write(int8_model_bytes)
    inspect_tflite(npu_int8_tflite_path, model_type)
    write_tflite_sidecar_manifest(npu_int8_tflite_path, model_type)

    # 임시 디렉터리 정리
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n==================================================")
    print(f"★ [PyTorch → Sony MCT 변환 성공] 모든 TFLite 아티팩트가 생성되었습니다.")
    print(f"  - Float32: {float_tflite_path}")
    print(f"  - Full INT8: {int8_tflite_path}")
    print(f"  - NPU INT8: {npu_int8_tflite_path}")
    print(f"==================================================")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoint to TFLite using Sony MCT")
    parser.add_argument("--pth-path", default="model/pytorch/best_crop_ir_mobilenetv3_fixed.pth", help="Path to PyTorch checkpoint")
    parser.add_argument("--output-prefix", default="model/pytorch/best_crop_ir_mobilenetv3_fixed", help="Prefix path for generated TFLite models")
    parser.add_argument("--model-type", choices=["crop_ir", "crop_rgb", "dual"], default="crop_ir", help="Model variant")
    parser.add_argument("--calib-samples", type=int, default=200, help="Number of calibration samples")
    parser.add_argument("--dataset-dir", default="dataset/raw/train", help="Dataset directory for calibration")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_pytorch_to_tflite(
        pth_path=args.pth_path,
        output_prefix=args.output_prefix,
        model_type=args.model_type,
        calib_samples=args.calib_samples,
        dataset_dir=args.dataset_dir
    )
