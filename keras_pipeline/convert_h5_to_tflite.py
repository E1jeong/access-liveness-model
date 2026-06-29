import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from keras_pipeline.tf_dataset import RGB_MEAN, RGB_STD, collect_items, representative_dataset
from keras_pipeline.tf_model import _rgb_current_norm_to_mobilenet_range


def _makedirs(path):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)


def convert_float(model_path, output_path):
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={
            "_rgb_current_norm_to_mobilenet_range": _rgb_current_norm_to_mobilenet_range,
        },
    )
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    _makedirs(output_path)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"[float tflite saved] {output_path}")


def convert_int8(model_path, output_path, data_dir, folds, fold_idx, seed, calibration_samples):
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={
            "_rgb_current_norm_to_mobilenet_range": _rgb_current_norm_to_mobilenet_range,
        },
    )
    train_items, _ = collect_items(data_dir, k_folds=folds, fold_idx=fold_idx, seed=seed)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(
        train_items,
        max_samples=calibration_samples,
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    _makedirs(output_path)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"[int8 tflite saved] {output_path}")


def _rgb_imagenet_norm_to_mobilenet_range(rgb):
    raw_0_1 = rgb * RGB_STD + RGB_MEAN
    return raw_0_1 * 2.0 - 1.0


def representative_dataset_npu(items, max_samples=200):
    from keras_pipeline.tf_dataset import load_sample

    for rgb_path, ir_path, _ in items[:max_samples]:
        rgb, ir = load_sample(rgb_path, ir_path, augment=False)
        rgb = _rgb_imagenet_norm_to_mobilenet_range(rgb)
        yield [
            np.expand_dims(rgb, axis=0).astype(np.float32),
            np.expand_dims(ir, axis=0).astype(np.float32),
        ]


def _copy_nested_weights(source_model, target_model, layer_name):
    source_layer = source_model.get_layer(layer_name)
    target_layer = target_model.get_layer(layer_name)
    for target_sub_layer in target_layer.layers:
        try:
            source_sub_layer = source_layer.get_layer(target_sub_layer.name)
        except ValueError:
            continue
        source_weights = source_sub_layer.get_weights()
        if source_weights:
            target_sub_layer.set_weights(source_weights)


def build_npu_export_model(trained_model):
    from keras_pipeline.tf_model import build_dual_mobilenetv2

    export_model = build_dual_mobilenetv2(
        rgb_weights=None,
        dropout=0.0,
        classifier_units=1024,
        ir_imagenet_init=False,
        rgb_input_mobilenet_range=True,
        average_pool_op=True,
        fixed_batch_size=1,
    )
    _copy_nested_weights(trained_model, export_model, "rgb_mobilenetv2")
    _copy_nested_weights(trained_model, export_model, "ir_mobilenetv2")
    for layer_name in ("classifier_dense", "logits"):
        export_model.get_layer(layer_name).set_weights(trained_model.get_layer(layer_name).get_weights())
    return export_model


def convert_int8_npu(model_path, output_path, data_dir, folds, fold_idx, seed, calibration_samples):
    trained_model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={
            "_rgb_current_norm_to_mobilenet_range": _rgb_current_norm_to_mobilenet_range,
        },
    )
    export_model = build_npu_export_model(trained_model)
    train_items, _ = collect_items(data_dir, k_folds=folds, fold_idx=fold_idx, seed=seed)

    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_npu(
        train_items,
        max_samples=calibration_samples,
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    _makedirs(output_path)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"[npu int8 tflite saved] {output_path}")


def inspect_tflite(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    print("[tflite tensors]")
    for idx, detail in enumerate(interpreter.get_input_details()):
        print(
            f" input {idx}: name={detail['name']} shape={detail['shape'].tolist()} "
            f"dtype={detail['dtype']} quant={detail['quantization']}"
        )
    for idx, detail in enumerate(interpreter.get_output_details()):
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
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-samples", type=int, default=500)
    parser.add_argument("--float", action="store_true", help="Write a float TFLite model.")
    parser.add_argument("--int8", action="store_true", help="Write a full INT8 TFLite model.")
    parser.add_argument("--npu-int8", action="store_true", help="Write an NNAPI/NPU-friendly full INT8 TFLite model.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.float and not args.int8 and not args.npu_int8:
        raise SystemExit("Choose at least one conversion mode: --float, --int8, and/or --npu-int8")
    if args.model_path is None:
        args.model_path = os.path.join(
            args.output_dir,
            f"best_model_fold{args.fold_idx}.keras",
        )
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(args.model_path)

    base_name = Path(args.model_path).stem
    if args.float:
        float_path = os.path.join(args.output_dir, f"{base_name}_float.tflite")
        convert_float(args.model_path, float_path)
        inspect_tflite(float_path)
    if args.int8:
        int8_path = os.path.join(args.output_dir, f"{base_name}_int8.tflite")
        convert_int8(
            args.model_path,
            int8_path,
            args.data_dir,
            args.folds,
            args.fold_idx,
            args.seed,
            args.calibration_samples,
        )
        inspect_tflite(int8_path)
    if args.npu_int8:
        npu_int8_path = os.path.join(args.output_dir, f"{base_name}_npu_int8.tflite")
        convert_int8_npu(
            args.model_path,
            npu_int8_path,
            args.data_dir,
            args.folds,
            args.fold_idx,
            args.seed,
            args.calibration_samples,
        )
        inspect_tflite(npu_int8_path)


if __name__ == "__main__":
    main()
