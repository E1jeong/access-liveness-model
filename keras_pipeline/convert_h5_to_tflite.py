import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from keras_pipeline.tf_dataset import collect_items, representative_dataset
from keras_pipeline.tf_model import _rgb_current_norm_to_mobilenet_range


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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"[int8 tflite saved] {output_path}")


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
        default="model/keras/best_model_fold0.keras",
    )
    parser.add_argument("--output-dir", default="model/keras")
    parser.add_argument("--data-dir", default="dataset/raw")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-samples", type=int, default=500)
    parser.add_argument("--float", action="store_true", help="Write a float TFLite model.")
    parser.add_argument("--int8", action="store_true", help="Write a full INT8 TFLite model.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.float and not args.int8:
        raise SystemExit("Choose at least one conversion mode: --float and/or --int8")
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


if __name__ == "__main__":
    main()
