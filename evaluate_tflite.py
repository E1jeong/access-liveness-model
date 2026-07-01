"""Evaluate float/int8 TFLite models on the validation split."""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


def _make_interpreter(model_path):
    try:
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
        is_litert = True
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        OpResolverType = None
        is_litert = False

    num_threads = os.cpu_count() or 4
    try:
        interp = Interpreter(model_path=model_path, num_threads=num_threads)
        interp.allocate_tensors()
        print(f"[interpreter] {'LiteRT' if is_litert else 'TF Lite'} XNNPACK path (num_threads={num_threads})")
        return interp
    except Exception:
        if is_litert and OpResolverType is not None:
            interp = Interpreter(
                model_path=model_path,
                num_threads=num_threads,
                experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
            )
        else:
            interp = Interpreter(model_path=model_path, num_threads=num_threads)
        interp.allocate_tensors()
        print(f"[interpreter] {'LiteRT' if is_litert else 'TF Lite'} reference kernel path (num_threads={num_threads})")
        return interp


def _quantize_input(arr, detail):
    dtype = detail["dtype"]
    if dtype == np.float32:
        return arr.astype(np.float32)
    scale, zero_point = detail["quantization"]
    q = np.round(arr / scale) + zero_point
    info = np.iinfo(dtype)
    return np.clip(q, info.min, info.max).astype(dtype)


def _dequantize_output(arr, detail):
    dtype = detail["dtype"]
    if dtype == np.float32:
        return arr.astype(np.float32)
    scale, zero_point = detail["quantization"]
    return (arr.astype(np.float32) - zero_point) * scale


def evaluate(model_path, data_dir, folds, fold_idx, seed, max_samples=None):
    from classes import CLASS_NAMES
    from keras_pipeline.tf_dataset import collect_items, load_multimodal_sample
    from utils import calculate_validation_metrics

    interp = _make_interpreter(model_path)
    in_details = interp.get_input_details()
    out_detail = interp.get_output_details()[0]

    def describe(detail):
        shape = [int(x) for x in detail["shape"]]
        if len(shape) == 4 and shape[1] in (1, 3):
            return "NCHW", shape[1]
        return "NHWC", shape[-1]

    def find_input(token, fallback_idx):
        token = token.lower()
        for detail in in_details:
            if token in detail["name"].lower():
                return detail
        return in_details[fallback_idx]

    input_specs = [
        ("cropRGB", find_input("crop_rgb", 0)),
        ("cropIR", find_input("crop_ir", 1)),
        ("RGB", find_input("c_rgb", 2)),
        ("IR", find_input("d_ir", 3)),
        ("heatmap", find_input("heatmap", 4)),
    ]
    layout_msg = ", ".join(f"{name}={describe(detail)[0]}" for name, detail in input_specs)
    print(f"[inputs] {layout_msg}")

    def build(sample_hwc, layout):
        batched = np.expand_dims(sample_hwc, axis=0).astype(np.float32)
        if layout == "NCHW":
            return np.transpose(batched, (0, 3, 1, 2))
        return batched

    _, val_items = collect_items(data_dir, k_folds=folds, fold_idx=fold_idx, seed=seed)
    total = len(val_items) if max_samples is None else min(len(val_items), max_samples)

    all_labels, all_preds = [], []
    pbar = tqdm(total=total, desc=f"evaluate {os.path.basename(model_path)}")
    for crop_rgb_path, crop_ir_path, label in val_items:
        sample = load_multimodal_sample(crop_rgb_path, crop_ir_path, augment=False)
        for sample_arr, (_, detail) in zip(sample, input_specs):
            layout, _ = describe(detail)
            arr = build(sample_arr, layout)
            interp.set_tensor(detail["index"], _quantize_input(arr, detail))

        interp.invoke()
        logits = _dequantize_output(interp.get_tensor(out_detail["index"]), out_detail)[0]
        all_labels.append(int(label))
        all_preds.append(int(np.argmax(logits)))
        pbar.update(1)

        if max_samples is not None and len(all_labels) >= max_samples:
            break
    pbar.close()

    cm, recalls, apcer, bpcer, acer = calculate_validation_metrics(all_labels, all_preds)
    acc = sum(int(l == p) for l, p in zip(all_labels, all_preds)) / len(all_labels)

    in_dtype = input_specs[0][1]["dtype"].__name__
    print(f"\n===== evaluation: {model_path} (input dtype={in_dtype}, samples={len(all_labels)}) =====")
    print(f" val_acc: {acc:.4f}")
    print(f" APCER: {apcer:.4f} | BPCER: {bpcer:.4f} | ACER: {acer:.4f}")
    print(" class recall:")
    for name, recall in zip(CLASS_NAMES, recalls):
        print(f"   {name}: {recall:.4f}")
    return {"model": model_path, "val_acc": acc, "apcer": apcer, "bpcer": bpcer, "acer": acer}


def main():
    parser = argparse.ArgumentParser(description="Evaluate float/int8 TFLite on the validation set")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["model/anti_spoofing.tflite", "model/anti_spoofing_float.tflite"],
        help="TFLite model paths to evaluate.",
    )
    parser.add_argument("--data-dir", default="dataset/raw")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    results = []
    for path in args.models:
        if not os.path.exists(path):
            print(f"[skip] missing model: {path}")
            continue
        results.append(evaluate(path, args.data_dir, args.folds, args.fold_idx, args.seed, args.max_samples))

    if len(results) > 1:
        print("\n===== model comparison =====")
        print(f"{'model':40s} {'val_acc':>8s} {'APCER':>8s} {'BPCER':>8s} {'ACER':>8s}")
        for result in results:
            print(
                f"{os.path.basename(result['model']):40s} "
                f"{result['val_acc']:8.4f} {result['apcer']:8.4f} "
                f"{result['bpcer']:8.4f} {result['acer']:8.4f}"
            )


if __name__ == "__main__":
    main()
