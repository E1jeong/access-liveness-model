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

from keras_pipeline.tf_dataset import (
    RGB_MEAN,
    RGB_STD,
    collect_items,
    load_sample,
    load_multimodal_sample,
)
from keras_pipeline.tf_model import _rgb_current_norm_to_mobilenet_range


def _makedirs(path):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)


def convert_float(model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    _makedirs(output_path)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"[float tflite saved] {output_path}")


def convert_int8(model, output_path, preloaded_samples, model_type):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def representative_dataset_gen():
        for sample in preloaded_samples:
            if model_type == "dual":
                yield [
                    np.expand_dims(sample[0], axis=0).astype(np.float32),
                    np.expand_dims(sample[1], axis=0).astype(np.float32),
                ]
            elif model_type == "crop_rgb":
                yield [
                    np.expand_dims(sample[0], axis=0).astype(np.float32),
                ]
            elif model_type == "crop_ir":
                yield [
                    np.expand_dims(sample[1], axis=0).astype(np.float32),
                ]
            else:
                yield {
                    "a_crop_rgb": np.expand_dims(sample[0], axis=0).astype(np.float32),
                    "b_crop_ir": np.expand_dims(sample[1], axis=0).astype(np.float32),
                    "c_rgb": np.expand_dims(sample[2], axis=0).astype(np.float32),
                    "d_ir": np.expand_dims(sample[3], axis=0).astype(np.float32),
                    "e_heatmap": np.expand_dims(sample[4], axis=0).astype(np.float32),
                }
            
    converter.representative_dataset = representative_dataset_gen
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
            ir_imagenet_init=False,
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
            ir_imagenet_init=False,
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

    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def representative_dataset_gen_npu():
        for sample in preloaded_samples:
            if model_type == "dual":
                rgb = _rgb_imagenet_norm_to_mobilenet_range(sample[0])
                yield [
                    np.expand_dims(rgb, axis=0).astype(np.float32),
                    np.expand_dims(sample[1], axis=0).astype(np.float32),
                ]
            elif model_type == "crop_rgb":
                rgb = _rgb_imagenet_norm_to_mobilenet_range(sample[0])
                yield [
                    np.expand_dims(rgb, axis=0).astype(np.float32),
                ]
            elif model_type == "crop_ir":
                yield [
                    np.expand_dims(sample[1], axis=0).astype(np.float32),
                ]
            else:
                crop_rgb = _rgb_imagenet_norm_to_mobilenet_range(sample[0])
                rgb = _rgb_imagenet_norm_to_mobilenet_range(sample[2])
                yield {
                    "a_crop_rgb": np.expand_dims(crop_rgb, axis=0).astype(np.float32),
                    "b_crop_ir": np.expand_dims(sample[1], axis=0).astype(np.float32),
                    "c_rgb": np.expand_dims(rgb, axis=0).astype(np.float32),
                    "d_ir": np.expand_dims(sample[3], axis=0).astype(np.float32),
                    "e_heatmap": np.expand_dims(sample[4], axis=0).astype(np.float32),
                }
            
    converter.representative_dataset = representative_dataset_gen_npu
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
    parser.add_argument(
        "--model-type",
        choices=["dual", "multimodal", "crop_rgb", "crop_ir"],
        default="dual",
        help="변환할 모델 종류 (dual: 2입력, multimodal: 5입력, crop_rgb: 단일 RGB, crop_ir: 단일 IR)"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-idx", type=int, default=0)
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
            filename = f"best_model_fold{args.fold_idx}.keras"
        elif args.model_type in ("crop_rgb", "crop_ir"):
            filename = f"best_{args.model_type}_fold{args.fold_idx}.keras"
        else:
            filename = f"best_multimodal_fold{args.fold_idx}.keras"
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

    preloaded_samples = None
    if args.int8 or args.npu_int8:
        train_items, _ = collect_items(
            args.data_dir,
            k_folds=args.folds,
            fold_idx=args.fold_idx,
            seed=args.seed,
        )
        # collect_items는 클래스 순서(live부터)로 쌓이므로 앞에서 자르면
        # 캘리브레이션이 live에 편향된다 — 전 클래스가 섞이도록 셔플 후 샘플링.
        random.Random(args.seed).shuffle(train_items)
        preloaded_samples = preload_calibration_samples(train_items, args.calibration_samples, args.model_type)

    base_name = Path(args.model_path).stem
    if args.float:
        float_path = os.path.join(args.output_dir, f"{base_name}_float.tflite")
        convert_float(model, float_path)
        inspect_tflite(float_path)
    if args.int8:
        int8_path = os.path.join(args.output_dir, f"{base_name}_int8.tflite")
        convert_int8(model, int8_path, preloaded_samples, args.model_type)
        inspect_tflite(int8_path)
    if args.npu_int8:
        npu_int8_path = os.path.join(args.output_dir, f"{base_name}_npu_int8.tflite")
        convert_int8_npu(model, npu_int8_path, preloaded_samples, args.model_type)
        inspect_tflite(npu_int8_path)


if __name__ == "__main__":
    main()
