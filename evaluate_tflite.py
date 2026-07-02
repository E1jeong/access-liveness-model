"""Evaluate float/int8 TFLite models on the validation split.

Run with .venv-tf (see docs/project_status.md §5). All --models are evaluated
in a single pass over the validation items: each sample is loaded once (with
threaded prefetch) and fed to every interpreter, so comparing N models costs
one dataset read instead of N.

NPU-friendly exports (filename contains "npu") take RGB in MobileNet [-1,1]
range instead of ImageNet normalization; --rgb-range auto handles this.
"""

import argparse
import collections
import concurrent.futures
import itertools
import os
import sys

import numpy as np
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# (logical name, tflite tensor-name token, fallback index in Keras input order)
INPUT_BINDINGS = [
    ("cropRGB", "crop_rgb", 0),
    ("cropIR", "crop_ir", 1),
    ("RGB", "c_rgb", 2),
    ("IR", "d_ir", 3),
    ("heatmap", "heatmap", 4),
]
# load_multimodal_sample 반환 순서에서 RGB 텐서 위치 (cropRGB, RGB)
RGB_INPUT_POSITIONS = (0, 2)


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


def _describe(detail):
    shape = [int(x) for x in detail["shape"]]
    if len(shape) == 4 and shape[1] in (1, 3):
        return "NCHW", shape[1]
    return "NHWC", shape[-1]


class TFLiteRunner:
    """One interpreter plus its input bindings and accumulated predictions."""

    def __init__(self, model_path, rgb_range, rgb_mean, rgb_std):
        self.model_path = model_path
        self.interp = _make_interpreter(model_path)
        in_details = self.interp.get_input_details()
        if len(in_details) != len(INPUT_BINDINGS):
            raise ValueError(
                f"{model_path}: expected {len(INPUT_BINDINGS)} inputs for the "
                f"5-input multimodal contract, got {len(in_details)}"
            )
        self.out_detail = self.interp.get_output_details()[0]

        def find_input(token, fallback_idx):
            for detail in in_details:
                if token in detail["name"].lower():
                    return detail
            return in_details[fallback_idx]

        self.input_specs = [
            (name, find_input(token, idx)) for name, token, idx in INPUT_BINDINGS
        ]
        indexes = [detail["index"] for _, detail in self.input_specs]
        if len(set(indexes)) != len(indexes):
            raise ValueError(
                f"{model_path}: input-name matching bound two logical inputs to "
                "the same tensor — check tensor names: "
                + ", ".join(d["name"] for d in in_details)
            )

        if rgb_range == "auto":
            self.rgb_mobilenet_range = "npu" in os.path.basename(model_path).lower()
        else:
            self.rgb_mobilenet_range = rgb_range == "mobilenet"
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

        layout_msg = ", ".join(
            f"{name}={_describe(detail)[0]}" for name, detail in self.input_specs
        )
        rgb_msg = "mobilenet [-1,1]" if self.rgb_mobilenet_range else "imagenet-normalized"
        print(f"[{os.path.basename(model_path)}] inputs: {layout_msg} | RGB range: {rgb_msg}")

        self.labels = []
        self.preds = []

    def run(self, sample, label):
        for pos, (sample_arr, (_, detail)) in enumerate(zip(sample, self.input_specs)):
            if self.rgb_mobilenet_range and pos in RGB_INPUT_POSITIONS:
                # load_multimodal_sample yields ImageNet-normalized RGB; NPU
                # exports expect MobileNet [-1,1] input (no in-graph Lambda).
                sample_arr = (sample_arr * self.rgb_std + self.rgb_mean) * 2.0 - 1.0
            arr = np.expand_dims(sample_arr, axis=0).astype(np.float32)
            if _describe(detail)[0] == "NCHW":
                arr = np.transpose(arr, (0, 3, 1, 2))
            self.interp.set_tensor(detail["index"], _quantize_input(arr, detail))

        self.interp.invoke()
        logits = _dequantize_output(self.interp.get_tensor(self.out_detail["index"]), self.out_detail)[0]
        self.labels.append(int(label))
        self.preds.append(int(np.argmax(logits)))


def _prefetched_samples(items, load_fn, num_workers, prefetch_depth):
    """Yield (item, loaded_sample) in order while loading ahead on threads."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        it = iter(items)
        pending = collections.deque(
            (item, executor.submit(load_fn, item))
            for item in itertools.islice(it, prefetch_depth)
        )
        while pending:
            item, future = pending.popleft()
            for nxt in itertools.islice(it, 1):
                pending.append((nxt, executor.submit(load_fn, nxt)))
            yield item, future.result()


def main():
    parser = argparse.ArgumentParser(description="Evaluate float/int8 TFLite on the validation set")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "model/keras/best_model_fold0_float.tflite",
            "model/keras/best_model_fold0_int8.tflite",
            "model/keras/best_model_fold0_npu_int8.tflite",
        ],
        help="TFLite model paths to evaluate (all run in one dataset pass).",
    )
    parser.add_argument("--data-dir", default="dataset/raw")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--rgb-range",
        choices=["auto", "imagenet", "mobilenet"],
        default="auto",
        help="RGB input range fed to each model. auto: MobileNet [-1,1] when the "
        "filename contains 'npu', ImageNet normalization otherwise.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Sample-loading threads.")
    args = parser.parse_args()

    from classes import CLASS_NAMES
    from keras_pipeline.tf_dataset import (
        RGB_MEAN,
        RGB_STD,
        collect_items,
        load_multimodal_sample,
    )
    from utils import calculate_validation_metrics

    runners = []
    for path in args.models:
        if not os.path.exists(path):
            print(f"[skip] missing model: {path}")
            continue
        runners.append(TFLiteRunner(path, args.rgb_range, RGB_MEAN, RGB_STD))
    if not runners:
        raise SystemExit("No model file found to evaluate.")

    _, val_items = collect_items(args.data_dir, k_folds=args.folds, fold_idx=args.fold_idx, seed=args.seed)
    if args.max_samples is not None:
        val_items = val_items[: args.max_samples]

    def load_fn(item):
        return load_multimodal_sample(item[0], item[1], augment=False)

    pbar = tqdm(total=len(val_items), desc=f"evaluate fold {args.fold_idx} ({len(runners)} models)")
    for item, sample in _prefetched_samples(
        val_items, load_fn, num_workers=args.num_workers, prefetch_depth=4 * args.num_workers
    ):
        for runner in runners:
            runner.run(sample, item[2])
        pbar.update(1)
    pbar.close()

    results = []
    for runner in runners:
        _, recalls, apcer, bpcer, acer = calculate_validation_metrics(runner.labels, runner.preds)
        acc = float(np.mean(np.asarray(runner.labels) == np.asarray(runner.preds)))
        in_dtype = runner.input_specs[0][1]["dtype"].__name__
        rgb_msg = "mobilenet [-1,1]" if runner.rgb_mobilenet_range else "imagenet-normalized"
        print(
            f"\n===== evaluation: {runner.model_path} "
            f"(input dtype={in_dtype}, RGB range={rgb_msg}, samples={len(runner.labels)}) ====="
        )
        print(f" val_acc: {acc:.4f}")
        print(f" APCER: {apcer:.4f} | BPCER: {bpcer:.4f} | ACER: {acer:.4f}")
        print(" class recall:")
        for name, recall in zip(CLASS_NAMES, recalls):
            print(f"   {name}: {recall:.4f}")
        results.append({"model": runner.model_path, "val_acc": acc, "apcer": apcer, "bpcer": bpcer, "acer": acer})

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
