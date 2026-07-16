import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import collect_split_items, validate_fixed_split_coverage
from classes import CLASS_NAMES
from keras_pipeline.model_signature import (
    validate_keras_model_signature,
    validate_tflite_model_signature,
)
from keras_pipeline.tf_dataset import (
    RGB_MEAN,
    RGB_STD,
    load_sample,
)
from keras_pipeline.tf_model import _rgb_current_norm_to_mobilenet_range
from keras_pipeline.artifact_paths import (
    keras_checkpoint_path,
    tflite_path as artifact_tflite_path,
    sidecar_manifest_path,
    calibration_manifest_path,
    check_no_overwrite,
)


def _makedirs(path):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)


def _validate_tflite_bytes(tflite_model, model_type):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    validate_tflite_model_signature(
        interpreter.get_input_details(), interpreter.get_output_details(), model_type
    )


def convert_float(model, output_path, model_type):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    _validate_tflite_bytes(tflite_model, model_type)
    _makedirs(output_path)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"[float tflite saved] {output_path}")


def _rgb_imagenet_norm_to_mobilenet_range(rgb):
    raw_0_1 = rgb * RGB_STD + RGB_MEAN
    return raw_0_1 * 2.0 - 1.0


def _load_calibration_sample(item, model_type):
    rgb_path, ir_path, _ = item
    return load_sample(rgb_path, ir_path, augment=False)


def _make_representative_dataset_gen(calibration_items, model_type, remap_rgb=False):
    """remap_rgb=True applies _rgb_imagenet_norm_to_mobilenet_range to the RGB
    channel(s) — used only for the NPU export graph, which has no in-graph
    Lambda to do this itself."""
    def _rgb(sample_channel):
        arr = _rgb_imagenet_norm_to_mobilenet_range(sample_channel) if remap_rgb else sample_channel
        return np.expand_dims(arr, axis=0).astype(np.float32)

    def gen():
        for item in calibration_items:
            sample = _load_calibration_sample(item, model_type)
            if model_type == "dual":
                yield [
                    _rgb(sample[0]),
                    np.expand_dims(sample[1], axis=0).astype(np.float32),
                ]
            elif model_type == "crop_rgb":
                yield [_rgb(sample[0])]
            else:
                yield [np.expand_dims(sample[1], axis=0).astype(np.float32)]

    return gen


def _convert_int8_core(keras_model, output_path, calibration_items, model_type, remap_rgb, log_label):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _make_representative_dataset_gen(calibration_items, model_type, remap_rgb)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    _validate_tflite_bytes(tflite_model, model_type)
    _makedirs(output_path)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"[{log_label} tflite saved] {output_path}")


def convert_int8(model, output_path, calibration_items, model_type):
    _convert_int8_core(model, output_path, calibration_items, model_type, remap_rgb=False, log_label="int8")


def _copy_nested_weights(source_model, target_model, layer_name):
    source_layer = source_model.get_layer(layer_name)
    target_layer = target_model.get_layer(layer_name)

    source_sub_layers = {layer.name: layer for layer in source_layer.layers}
    target_weighted_layers = [layer for layer in target_layer.layers if layer.get_weights()]
    source_weighted_names = {
        layer.name for layer in source_layer.layers if layer.get_weights()
    }
    target_weighted_names = {layer.name for layer in target_weighted_layers}
    missing_in_target = source_weighted_names - target_weighted_names
    if missing_in_target:
        raise ValueError(
            f"{layer_name}: export backbone에 없는 source weighted layer: "
            f"{sorted(missing_in_target)}"
        )

    copied = []
    for target_sub_layer in target_layer.layers:
        target_weights = target_sub_layer.get_weights()
        if not target_weights:
            continue

        source_sub_layer = source_sub_layers.get(target_sub_layer.name)
        if source_sub_layer is None:
            raise ValueError(
                f"{layer_name}: source backbone에 없는 export weighted layer: "
                f"{target_sub_layer.name}"
            )
        source_weights = source_sub_layer.get_weights()
        source_shapes = [tuple(weight.shape) for weight in source_weights]
        target_shapes = [tuple(weight.shape) for weight in target_weights]
        if source_shapes != target_shapes:
            raise ValueError(
                f"{layer_name}/{target_sub_layer.name}: weight tensor shape mismatch "
                f"(source={source_shapes}, target={target_shapes})"
            )
        target_sub_layer.set_weights(source_weights)
        copied.append((target_sub_layer.name, source_shapes))

    print(f"[weight copy] {layer_name}: {copied}")


def _npu_parity_inputs(sample, model_type):
    def batch(value):
        return np.expand_dims(value, axis=0).astype(np.float32)

    if model_type == "dual":
        return [batch(sample[0]), batch(sample[1])], [
            batch(_rgb_imagenet_norm_to_mobilenet_range(sample[0])),
            batch(sample[1]),
        ]
    if model_type == "crop_rgb":
        return [batch(sample[0])], [batch(_rgb_imagenet_norm_to_mobilenet_range(sample[0]))]
    if model_type == "crop_ir":
        return [batch(sample[1])], [batch(sample[1])]

    source_inputs = [batch(value) for value in sample]
    export_inputs = [
        batch(_rgb_imagenet_norm_to_mobilenet_range(sample[0])),
        batch(sample[1]),
        batch(_rgb_imagenet_norm_to_mobilenet_range(sample[2])),
        batch(sample[3]),
        batch(sample[4]),
    ]
    return source_inputs, export_inputs


def validate_npu_export_parity(trained_model, export_model, sample, model_type):
    """Fail when the NPU export graph no longer preserves Keras logits."""
    source_inputs, export_inputs = _npu_parity_inputs(sample, model_type)
    source_call_inputs = source_inputs[0] if len(source_inputs) == 1 else source_inputs
    export_call_inputs = export_inputs[0] if len(export_inputs) == 1 else export_inputs
    trained_logits = trained_model(source_call_inputs, training=False).numpy()
    export_logits = export_model(export_call_inputs, training=False).numpy()
    error = np.abs(trained_logits - export_logits)
    result = {
        "max_error": float(error.max()),
        "mean_error": float(error.mean()),
        "argmax_agreement": bool(np.array_equal(
            np.argmax(trained_logits, axis=-1), np.argmax(export_logits, axis=-1)
        )),
    }
    if not result["argmax_agreement"] or not np.allclose(
        trained_logits, export_logits, rtol=1e-5, atol=1e-5
    ):
        raise ValueError(f"NPU export Keras logits parity failed: {result}")
    print(f"[NPU export Keras logits parity] {result}")
    return result


def build_npu_export_model(trained_model, model_type):
    # 학습 체크포인트에서 classifier 구성을 읽는다 (--classifier-units 0 지원).
    try:
        trained_dense = trained_model.get_layer("classifier_dense")
        classifier_units = trained_dense.units
    except ValueError:
        trained_dense = None
        classifier_units = 0

    if model_type == "dual":
        from keras_pipeline.tf_model import build_dual_mobilenetv2
        export_model = build_dual_mobilenetv2(
            rgb_weights=None,
            dropout=0.0,
            classifier_units=classifier_units,
            gray_imagenet_init=False,
            rgb_input_mobilenet_range=True,
            average_pool_op=True,
            fixed_batch_size=1,
            classifier_as_conv=True,
        )
        backbones = ["rgb_mobilenetv2", "ir_mobilenetv2"]
    else:
        from keras_pipeline.tf_model import build_single_mobilenetv2
        export_model = build_single_mobilenetv2(
            input_type=model_type,
            rgb_weights=None,
            dropout=0.0,
            classifier_units=classifier_units,
            gray_imagenet_init=False,
            rgb_input_mobilenet_range=True,
            average_pool_op=True,
            fixed_batch_size=1,
            classifier_as_conv=True,
        )
        backbones = [f"{model_type}_mobilenetv2"]

    for layer_name in backbones:
        _copy_nested_weights(trained_model, export_model, layer_name)

    # classifier_dense (Dense) -> classifier_dense_conv (Conv2D 1x1)
    if trained_dense is not None:
        export_dense_conv = export_model.get_layer("classifier_dense_conv")
        dense_w, dense_b = trained_dense.get_weights()
        conv_w = np.reshape(dense_w, (1, 1, dense_w.shape[0], dense_w.shape[1]))
        export_dense_conv.set_weights([conv_w, dense_b])

    # logits (Dense) -> logits_conv (Conv2D 1x1)
    trained_logits = trained_model.get_layer("logits")
    export_logits_conv = export_model.get_layer("logits_conv")
    logits_w, logits_b = trained_logits.get_weights()
    logits_conv_w = np.reshape(logits_w, (1, 1, logits_w.shape[0], logits_w.shape[1]))
    export_logits_conv.set_weights([logits_conv_w, logits_b])

    return export_model


def convert_int8_npu(trained_model, output_path, calibration_items, model_type):
    export_model = build_npu_export_model(trained_model, model_type)
    if not calibration_items:
        raise ValueError("NPU export Keras logits parity requires at least one calibration sample")
    validate_npu_export_parity(
        trained_model,
        export_model,
        _load_calibration_sample(calibration_items[0], model_type),
        model_type,
    )
    _convert_int8_core(export_model, output_path, calibration_items, model_type, remap_rgb=True, log_label="npu int8")


def inspect_tflite(path, model_type):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    validate_tflite_model_signature(input_details, output_details, model_type)
    print("[tflite tensors]")
    for idx, detail in enumerate(input_details):
        print(
            f" input {idx}: name={detail['name']} shape={detail['shape'].tolist()} "
            f"dtype={detail['dtype']} quant={detail['quantization']}"
        )
    for idx, detail in enumerate(output_details):
        print(
            f" output {idx}: name={detail['name']} shape={detail['shape'].tolist()} "
            f"dtype={detail['dtype']} quant={detail['quantization']}"
        )


def write_tflite_sidecar_manifest(tflite_path, model_type):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    is_npu_int8 = "npu_int8" in os.path.basename(tflite_path)

    inputs_info = []
    for idx, detail in enumerate(input_details):
        name = detail['name']
        shape = detail['shape'].tolist()
        dtype = detail['dtype'].__name__
        channels = shape[-1]

        # input_kind 결정
        input_kind = "unknown"
        if model_type == "crop_rgb":
            input_kind = "rgb"
        elif model_type == "crop_ir":
            input_kind = "ir"
        elif model_type == "dual":
            input_kind = "rgb" if channels == 3 else "ir"

        # Quantization 파라미터
        scale, zero_point = detail['quantization']
        quant = None
        if scale != 0.0 or zero_point != 0:
            quant = {
                "scale": float(scale),
                "zero_point": int(zero_point)
            }

        # Normalization 결정
        if channels == 3:
            if is_npu_int8:
                norm = {
                    "mean": [0.5, 0.5, 0.5],
                    "std": [0.5, 0.5, 0.5],
                    "range": "[-1, 1]"
                }
            else:
                norm = {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "range": "imagenet"
                }
        else:  # channels == 1
            norm = {
                "mean": [0.5],
                "std": [0.5],
                "range": "[-1, 1]"
            }

        inputs_info.append({
            "name": name,
            "index": idx,
            "shape": shape,
            "dtype": dtype,
            "input_kind": input_kind,
            "quantization": quant,
            "normalization": norm
        })

    outputs_info = []
    for idx, detail in enumerate(output_details):
        name = detail['name']
        shape = detail['shape'].tolist()
        dtype = detail['dtype'].__name__

        scale, zero_point = detail['quantization']
        quant = None
        if scale != 0.0 or zero_point != 0:
            quant = {
                "scale": float(scale),
                "zero_point": int(zero_point)
            }

        outputs_info.append({
            "name": name,
            "index": idx,
            "shape": shape,
            "dtype": dtype,
            "quantization": quant,
            "output_is_logits": True
        })

    from classes import CLASS_NAMES
    manifest = {
        "model_type": model_type,
        "file_name": os.path.basename(tflite_path),
        "delegate": "nnapi" if is_npu_int8 else "cpu",
        "inputs": inputs_info,
        "outputs": outputs_info,
        "class_order": CLASS_NAMES,
        "crop_margin_ratio": 0.10
    }

    manifest_path = tflite_path.replace(".tflite", "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[tflite sidecar manifest saved] {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a saved Keras model to TFLite.")
    parser.add_argument(
        "--model-path",
        "--h5-path",
        dest="model_path",
        default=None,
    )
    parser.add_argument("--output-dir", default="model/keras")
    parser.add_argument("--data-dir", default="dataset/raw")
    parser.add_argument(
        "--model-type",
        choices=["dual", "crop_rgb", "crop_ir"],
        default="dual",
        help="변환할 모델 종류 (dual: 2입력, crop_rgb: 단일 RGB, crop_ir: 단일 IR)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-samples", type=int, default=500)
    parser.add_argument("--float", action="store_true", help="Write a float TFLite model.")
    parser.add_argument("--int8", action="store_true", help="Write a full INT8 TFLite model.")
    parser.add_argument("--npu-int8", action="store_true", help="Write an NNAPI/NPU-friendly full INT8 TFLite model.")
    parser.add_argument("--force", action="store_true", help="기존 산출물을 덮어쓰기 허용")
    return parser.parse_args()


def select_stratified_calibration_items(items, max_samples, seed):
    if max_samples < len(CLASS_NAMES):
        raise ValueError(
            f"calibration samples는 모든 {len(CLASS_NAMES)} 클래스를 포함해야 합니다: {max_samples}"
        )

    by_label = {label: [] for label in range(len(CLASS_NAMES))}
    for item in items:
        label = item[2]
        if label not in by_label:
            raise ValueError(f"알 수 없는 calibration label: {label}")
        by_label[label].append(item)

    missing_labels = [CLASS_NAMES[label] for label, values in by_label.items() if not values]
    if missing_labels:
        raise ValueError(f"train calibration에 없는 클래스: {missing_labels}")

    rng = random.Random(seed)
    labels = list(by_label)
    rng.shuffle(labels)
    for values in by_label.values():
        rng.shuffle(values)

    selected = []
    positions = {label: 0 for label in labels}
    while len(selected) < max_samples:
        added = False
        for label in labels:
            if positions[label] >= len(by_label[label]):
                continue
            selected.append(by_label[label][positions[label]])
            positions[label] += 1
            added = True
            if len(selected) == max_samples:
                break
        if not added:
            break

    selected_by_label = {label: 0 for label in by_label}
    for _, _, label in selected:
        selected_by_label[label] += 1
    if any(count == 0 for count in selected_by_label.values()):
        raise ValueError("stratified calibration이 모든 클래스를 포함하지 못했습니다")

    return selected, {
        "available_by_class": {
            CLASS_NAMES[label]: len(by_label[label]) for label in by_label
        },
        "selected_by_class": {
            CLASS_NAMES[label]: selected_by_label[label] for label in selected_by_label
        },
    }


def collect_calibration_items(data_dir, max_samples, seed):
    """Collect calibration inputs exclusively from the fixed train split."""
    return select_stratified_calibration_items(
        collect_split_items(data_dir, "train"), max_samples, seed
    )


def _estimated_calibration_sample_bytes(model_type):
    channels = {"dual": 4, "crop_rgb": 3, "crop_ir": 1}[model_type]
    return 224 * 224 * channels * np.dtype(np.float32).itemsize


def write_calibration_manifest(path, items, report, model_type, seed, requested_samples):
    payload = {
        "split": "train",
        "model_type": model_type,
        "seed": seed,
        "requested_samples": requested_samples,
        "selected_samples": len(items),
        **report,
        "missing_required_samples": 0,
        "estimated_peak_sample_bytes": _estimated_calibration_sample_bytes(model_type),
        "preloaded_sample_bytes": 0,
        "items": [
            {"crop_rgb_path": rgb_path, "crop_ir_path": ir_path, "label": label}
            for rgb_path, ir_path, label in items
        ],
    }
    _makedirs(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[calibration manifest] {path}: {payload['selected_by_class']}, "
          f"peak~{payload['estimated_peak_sample_bytes']} bytes, preload=0 bytes")


def main():
    args = parse_args()
    if not args.float and not args.int8 and not args.npu_int8:
        raise SystemExit("Choose at least one conversion mode: --float, --int8, and/or --npu-int8")
    if args.model_path is None:
        args.model_path = keras_checkpoint_path(args.output_dir, args.model_type)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(args.model_path)

    print(f"Loading trained Keras model: {args.model_path}")
    model = tf.keras.models.load_model(
        args.model_path,
        compile=False,
        custom_objects={
            "_rgb_current_norm_to_mobilenet_range": _rgb_current_norm_to_mobilenet_range,
        },
    )
    validate_keras_model_signature(model, args.model_type)

    calibration_items = None
    if args.int8 or args.npu_int8:
        validate_fixed_split_coverage(args.data_dir)
        print("[calibration split] train only")
        calibration_items, calibration_report = collect_calibration_items(
            args.data_dir, args.calibration_samples, args.seed
        )
        manifest_path = os.path.join(
            args.output_dir, f"{Path(args.model_path).stem}_calibration_manifest.json"
        )
        write_calibration_manifest(
            manifest_path,
            calibration_items,
            calibration_report,
            args.model_type,
            args.seed,
            args.calibration_samples,
        )

    base_name = Path(args.model_path).stem
    if args.float:
        float_path = artifact_tflite_path(args.output_dir, base_name, "float")
        check_no_overwrite(float_path, force=args.force)
        convert_float(model, float_path, args.model_type)
        inspect_tflite(float_path, args.model_type)
        write_tflite_sidecar_manifest(float_path, args.model_type)
    if args.int8:
        int8_path = artifact_tflite_path(args.output_dir, base_name, "int8")
        check_no_overwrite(int8_path, force=args.force)
        convert_int8(model, int8_path, calibration_items, args.model_type)
        inspect_tflite(int8_path, args.model_type)
        write_tflite_sidecar_manifest(int8_path, args.model_type)
    if args.npu_int8:
        npu_int8_path = artifact_tflite_path(args.output_dir, base_name, "npu_int8")
        check_no_overwrite(npu_int8_path, force=args.force)
        convert_int8_npu(model, npu_int8_path, calibration_items, args.model_type)
        inspect_tflite(npu_int8_path, args.model_type)
        write_tflite_sidecar_manifest(npu_int8_path, args.model_type)


if __name__ == "__main__":
    main()
