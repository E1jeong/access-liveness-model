import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import concurrent.futures

from utils import collect_split_items, validate_fixed_split_coverage
from keras_pipeline.model_signature import (
    validate_keras_model_signature,
    validate_tflite_model_signature,
)
from keras_pipeline.tf_dataset import (
    RGB_MEAN,
    RGB_STD,
    load_sample,
    load_multimodal_sample,
)
from keras_pipeline.tf_model import _rgb_current_norm_to_mobilenet_range


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


def _make_representative_dataset_gen(preloaded_samples, model_type, remap_rgb=False):
    """remap_rgb=True applies _rgb_imagenet_norm_to_mobilenet_range to the RGB
    channel(s) — used only for the NPU export graph, which has no in-graph
    Lambda to do this itself."""
    def _rgb(sample_channel):
        arr = _rgb_imagenet_norm_to_mobilenet_range(sample_channel) if remap_rgb else sample_channel
        return np.expand_dims(arr, axis=0).astype(np.float32)

    def gen():
        for sample in preloaded_samples:
            if model_type == "dual":
                yield [
                    _rgb(sample[0]),
                    np.expand_dims(sample[1], axis=0).astype(np.float32),
                ]
            elif model_type == "crop_rgb":
                yield [_rgb(sample[0])]
            elif model_type == "crop_ir":
                yield [np.expand_dims(sample[1], axis=0).astype(np.float32)]
            else:
                yield {
                    "a_crop_rgb": _rgb(sample[0]),
                    "b_crop_ir": np.expand_dims(sample[1], axis=0).astype(np.float32),
                    "c_rgb": _rgb(sample[2]),
                    "d_ir": np.expand_dims(sample[3], axis=0).astype(np.float32),
                    "e_heatmap": np.expand_dims(sample[4], axis=0).astype(np.float32),
                }

    return gen


def _convert_int8_core(keras_model, output_path, preloaded_samples, model_type, remap_rgb, log_label):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _make_representative_dataset_gen(preloaded_samples, model_type, remap_rgb)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    _validate_tflite_bytes(tflite_model, model_type)
    _makedirs(output_path)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"[{log_label} tflite saved] {output_path}")


def convert_int8(model, output_path, preloaded_samples, model_type):
    _convert_int8_core(model, output_path, preloaded_samples, model_type, remap_rgb=False, log_label="int8")


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
    elif model_type in ("crop_rgb", "crop_ir"):
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
    else:
        from keras_pipeline.tf_model import build_multimodal_mobilenetv2
        export_model = build_multimodal_mobilenetv2(
            rgb_weights=None,
            dropout=0.0,
            classifier_units=classifier_units,
            gray_imagenet_init=False,
            rgb_input_mobilenet_range=True,
            average_pool_op=True,
            fixed_batch_size=1,
            classifier_as_conv=True,
        )
        backbones = [
            "crop_rgb_mobilenetv2",
            "crop_ir_mobilenetv2",
            "rgb_mobilenetv2",
            "ir_mobilenetv2",
            "heatmap_mobilenetv2",
        ]

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


def convert_int8_npu(trained_model, output_path, preloaded_samples, model_type):
    export_model = build_npu_export_model(trained_model, model_type)
    if not preloaded_samples:
        raise ValueError("NPU export Keras logits parity requires at least one calibration sample")
    validate_npu_export_parity(
        trained_model, export_model, preloaded_samples[0], model_type
    )
    _convert_int8_core(export_model, output_path, preloaded_samples, model_type, remap_rgb=True, log_label="npu int8")


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
        choices=["dual", "multimodal", "crop_rgb", "crop_ir"],
        default="dual",
        help="변환할 모델 종류 (dual: 2입력, multimodal: 5입력, crop_rgb: 단일 RGB, crop_ir: 단일 IR)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-samples", type=int, default=500)
    parser.add_argument("--float", action="store_true", help="Write a float TFLite model.")
    parser.add_argument("--int8", action="store_true", help="Write a full INT8 TFLite model.")
    parser.add_argument("--npu-int8", action="store_true", help="Write an NNAPI/NPU-friendly full INT8 TFLite model.")
    return parser.parse_args()


def preload_calibration_samples(items, max_samples, model_type):
    count = min(len(items), max_samples)
    print(f"Preloading {count} calibration samples in parallel (mode: {model_type})...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        if model_type in ("dual", "crop_rgb", "crop_ir"):
            def worker(item):
                rgb_path, ir_path, _ = item
                return load_sample(rgb_path, ir_path, augment=False)
        else:
            def worker(item):
                crop_rgb_path, crop_ir_path, _ = item
                return load_multimodal_sample(crop_rgb_path, crop_ir_path, augment=False)
        futures = executor.map(worker, items[:max_samples])
        loaded = list(futures)
    return loaded


def main():
    args = parse_args()
    if not args.float and not args.int8 and not args.npu_int8:
        raise SystemExit("Choose at least one conversion mode: --float, --int8, and/or --npu-int8")
    if args.model_path is None:
        if args.model_type == "dual":
            filename = "best_model_fixed.keras"
        elif args.model_type in ("crop_rgb", "crop_ir"):
            filename = f"best_{args.model_type}_fixed.keras"
        else:
            filename = "best_multimodal_fixed.keras"
        args.model_path = os.path.join(
            args.output_dir,
            filename,
        )
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

    preloaded_samples = None
    if args.int8 or args.npu_int8:
        validate_fixed_split_coverage(args.data_dir)
        train_items = collect_split_items(args.data_dir, "train")
        # collect_split_items는 클래스 순서(live부터)로 쌓이므로 앞에서 자르면
        # 캘리브레이션이 live에 편향된다 — 전 클래스가 섞이도록 셔플 후 샘플링.
        random.Random(args.seed).shuffle(train_items)
        print("[calibration split] train only")
        preloaded_samples = preload_calibration_samples(train_items, args.calibration_samples, args.model_type)

    base_name = Path(args.model_path).stem
    if args.float:
        float_path = os.path.join(args.output_dir, f"{base_name}_float.tflite")
        convert_float(model, float_path, args.model_type)
        inspect_tflite(float_path, args.model_type)
    if args.int8:
        int8_path = os.path.join(args.output_dir, f"{base_name}_int8.tflite")
        convert_int8(model, int8_path, preloaded_samples, args.model_type)
        inspect_tflite(int8_path, args.model_type)
    if args.npu_int8:
        npu_int8_path = os.path.join(args.output_dir, f"{base_name}_npu_int8.tflite")
        convert_int8_npu(model, npu_int8_path, preloaded_samples, args.model_type)
        inspect_tflite(npu_int8_path, args.model_type)


if __name__ == "__main__":
    main()
