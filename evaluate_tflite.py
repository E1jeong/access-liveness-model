"""학습한 TFLite 모델을 고정 validation 또는 test split으로 평가합니다.

INT8 양자화 후에도 보안 지표(APCER/BPCER/ACER)가 유지되는지 확인하는 용도.
입력은 학습과 동일하게 정규화한 뒤, 모델이 int8이면 scale/zero_point로 양자화하고,
출력이 int8이면 역양자화하여 float 모델과 동일 기준으로 지표를 계산합니다.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def _make_interpreter(model_path):
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    n = os.cpu_count() or 4
    try:
        interp = Interpreter(model_path=model_path, num_threads=n)
        interp.allocate_tensors()
        print(f"[interpreter] XNNPACK 경로 (num_threads={n})")
        return interp
    except Exception:
        interp = Interpreter(
            model_path=model_path,
            num_threads=n,
            experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        interp.allocate_tensors()
        print(f"[interpreter] reference 커널 경로 (XNNPACK 미사용, num_threads={n})")
        return interp


def _summarize_result(name, model_path, labels, preds, logits, latencies_ms, file_size_bytes=None):
    from utils import calculate_validation_metrics

    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)
    _, recalls, apcer, bpcer, acer = calculate_validation_metrics(labels, preds)
    return {
        "name": name,
        "model": model_path,
        "accuracy": float(np.mean(labels == preds)),
        "apcer": float(apcer),
        "bpcer": float(bpcer),
        "acer": float(acer),
        "recalls": [float(value) for value in recalls],
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "file_size_bytes": file_size_bytes,
        "_labels": labels,
        "_preds": preds,
        "_logits": logits,
    }


def write_regression_report(results, output_path, split="validation"):
    if len(results) < 2:
        raise ValueError("artifact regression report에는 비교할 결과가 두 개 이상 필요합니다")

    baseline = results[0]
    comparisons = []
    for result in results[1:]:
        if not np.array_equal(baseline["_labels"], result["_labels"]):
            raise ValueError(f"비교 결과의 label 순서가 다릅니다: {result['name']}")
        error = np.abs(baseline["_logits"] - result["_logits"])
        comparisons.append({
            "artifact": result["name"],
            "logits_max_abs_error": float(error.max()),
            "logits_mean_abs_error": float(error.mean()),
            "argmax_agreement": float(np.mean(
                baseline["_preds"] == result["_preds"]
            )),
            "acer_delta": float(result["acer"] - baseline["acer"]),
            "mean_latency_ms_delta": float(
                result["mean_latency_ms"] - baseline["mean_latency_ms"]
            ),
        })

    report = {
        "split": split,
        "baseline": baseline["name"],
        "artifacts": [{
            key: value for key, value in result.items() if not key.startswith("_")
        } for result in results],
        "comparisons": comparisons,
        "acer_policy": "report_only",
    }
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[artifact regression report] {output_path}")
    return report


def write_metrics_csv(results, output_path, split):
    fields = ["split", "name", "model", "accuracy", "apcer", "bpcer", "acer", "mean_latency_ms", "file_size_bytes"]
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key) for key in fields if key != "split"} | {"split": split})
    print(f"[metrics csv] {output_path}")


def evaluate(model_path, data_dir, split, model_type, max_samples=None):
    from keras_pipeline.tf_dataset import (
        load_sample,
        RGB_MEAN,
        RGB_STD,
    )
    from utils import (
        calculate_validation_metrics,
        collect_split_items,
        dequantize_from_tflite,
        quantize_for_tflite,
        validate_fixed_split_coverage,
    )
    from classes import CLASS_NAMES

    is_npu_int8 = "npu_int8" in os.path.basename(model_path)

    interp = _make_interpreter(model_path)
    in_details = interp.get_input_details()
    out_detail = interp.get_output_details()[0]
    output_size = int(out_detail["shape"][-1])
    if output_size != len(CLASS_NAMES):
        raise ValueError(
            f"Model output size is {output_size}, expected {len(CLASS_NAMES)} "
            f"for classes: {CLASS_NAMES}"
        )

    def describe(d):
        shape = [int(x) for x in d['shape']]
        if len(shape) == 4 and shape[1] in (1, 3):
            return "NCHW", shape[1]
        return "NHWC", shape[-1]

    metas = [(d, *describe(d)) for d in in_details]

    rgb_d, rgb_layout = None, None
    ir_d, ir_layout = None, None

    if model_type == "dual":
        rgb_d, rgb_layout, _ = next(m for m in metas if m[2] == 3)
        ir_d, ir_layout, _ = next(m for m in metas if m[2] == 1)
        print(f" 입력 레이아웃: rgb={rgb_layout}, ir={ir_layout}")

    elif model_type == "crop_rgb":
        rgb_d, rgb_layout, _ = next(m for m in metas if m[2] == 3)
        print(f" 입력 레이아웃: rgb={rgb_layout}")
    elif model_type == "crop_ir":
        ir_d, ir_layout, _ = next(m for m in metas if m[2] == 1)
        print(f" 입력 레이아웃: ir={ir_layout}")

    def build(sample_hwc, layout):
        if layout == "NCHW":
            sample_hwc = np.transpose(sample_hwc, (2, 0, 1))
        return np.expand_dims(sample_hwc, axis=0).astype(np.float32)

    split_counts = validate_fixed_split_coverage(data_dir)
    eval_items = collect_split_items(data_dir, split)
    print(
        f"[데이터셋 구성 완료] split={split}, 평가 데이터 수: {len(eval_items)}장 "
        f"(train={split_counts['train']}, validation={split_counts['validation']}, "
        f"test={split_counts['test']})"
    )

    all_labels, all_preds, all_logits, latencies_ms = [], [], [], []
    total = len(eval_items)
    if max_samples is not None:
        total = min(total, max_samples)
    pbar = tqdm(total=total, desc=f"평가 {os.path.basename(model_path)}")
    for rgb_path, ir_path, label in eval_items[:total]:
        rgb_hwc, ir_hwc = load_sample(rgb_path, ir_path, augment=False)
        if is_npu_int8:
            rgb_hwc = (rgb_hwc * RGB_STD + RGB_MEAN) * 2.0 - 1.0

        if model_type == "dual":
            interp.set_tensor(rgb_d['index'], quantize_for_tflite(build(rgb_hwc, rgb_layout), rgb_d))
            interp.set_tensor(ir_d['index'], quantize_for_tflite(build(ir_hwc, ir_layout), ir_d))
        elif model_type == "crop_rgb":
            interp.set_tensor(rgb_d['index'], quantize_for_tflite(build(rgb_hwc, rgb_layout), rgb_d))
        elif model_type == "crop_ir":
            interp.set_tensor(ir_d['index'], quantize_for_tflite(build(ir_hwc, ir_layout), ir_d))

        start = time.perf_counter()
        interp.invoke()
        latencies_ms.append((time.perf_counter() - start) * 1000)
        logits = dequantize_from_tflite(interp.get_tensor(out_detail['index']), out_detail)[0]
        all_labels.append(int(label))
        all_preds.append(int(np.argmax(logits)))
        all_logits.append(logits)
        pbar.update(1)
    pbar.close()

    result = _summarize_result(
        os.path.basename(model_path),
        model_path,
        all_labels,
        all_preds,
        all_logits,
        latencies_ms,
        os.path.getsize(model_path),
    )

    active_d = rgb_d if rgb_d is not None else ir_d
    in_dtype = active_d['dtype'].__name__
    print(f"\n===== 평가: {model_path} (split={split}, 입력 dtype={in_dtype}, {len(all_labels)}장) =====")
    print(f" accuracy: {result['accuracy']:.4f}")
    print(f" APCER: {result['apcer']:.4f} | BPCER: {result['bpcer']:.4f} | ACER: {result['acer']:.4f}")
    print(f" mean latency: {result['mean_latency_ms']:.3f} ms")
    print(" 클래스별 Recall:")
    for name, r in zip(CLASS_NAMES, result["recalls"]):
        print(f"   {name}: {r:.4f}")
    return result


def evaluate_keras_model(model_path, data_dir, split, model_type, max_samples=None, npu_export=False):
    import tensorflow as tf
    from keras_pipeline.convert_keras_to_tflite import build_npu_export_model
    from keras_pipeline.model_signature import validate_keras_model_signature
    from keras_pipeline.tf_dataset import load_sample, RGB_MEAN, RGB_STD
    from keras_pipeline.tf_model import _rgb_current_norm_to_mobilenet_range
    from utils import collect_split_items, validate_fixed_split_coverage

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"_rgb_current_norm_to_mobilenet_range": _rgb_current_norm_to_mobilenet_range},
    )
    validate_keras_model_signature(model, model_type)
    name = "npu_export_keras" if npu_export else "keras"
    if npu_export:
        model = build_npu_export_model(model, model_type)

    validate_fixed_split_coverage(data_dir)
    items = collect_split_items(data_dir, split)
    if max_samples is not None:
        items = items[:max_samples]

    labels, preds, logits_list, latencies_ms = [], [], [], []
    for rgb_path, ir_path, label in tqdm(items, desc=f"평가 {name}"):
        rgb, ir = load_sample(rgb_path, ir_path, augment=False)
        if npu_export:
            rgb = (rgb * RGB_STD + RGB_MEAN) * 2.0 - 1.0
        if model_type == "dual":
            inputs = [np.expand_dims(rgb, axis=0), np.expand_dims(ir, axis=0)]
        elif model_type == "crop_rgb":
            inputs = np.expand_dims(rgb, axis=0)
        else:
            inputs = np.expand_dims(ir, axis=0)

        start = time.perf_counter()
        logits = model(inputs, training=False).numpy()[0]
        latencies_ms.append((time.perf_counter() - start) * 1000)
        labels.append(int(label))
        preds.append(int(np.argmax(logits)))
        logits_list.append(logits)

    return _summarize_result(name, model_path, labels, preds, logits_list, latencies_ms)


def main():
    parser = argparse.ArgumentParser(description="Evaluate float/int8 TFLite on a fixed split")
    parser.add_argument("--models", nargs="+", default=[], help="평가할 TFLite 경로들")
    parser.add_argument("--keras-model", help="원본 Keras checkpoint 경로")
    parser.add_argument("--npu-export", action="store_true", help="Keras checkpoint에서 동적 NPU export Keras도 비교")
    parser.add_argument("--report-json", help="artifact regression JSON 출력 경로")
    parser.add_argument("--report-csv", help="평가 metric CSV 출력 경로 (test는 명시적으로만 실행)")
    parser.add_argument("--data-dir", default="dataset/raw")
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="기본은 개발용 validation. 설정 확정 후 최종 평가에만 test를 명시한다.",
    )
    parser.add_argument(
        "--model-type",
        choices=["dual", "crop_rgb", "crop_ir"],
        default="dual",
        help="평가할 모델 종류"
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if args.npu_export and not args.keras_model:
        raise SystemExit("--npu-export에는 --keras-model이 필요합니다")

    results = []
    if args.keras_model:
        results.append(evaluate_keras_model(
            args.keras_model, args.data_dir, args.split, args.model_type, args.max_samples
        ))
        if args.npu_export:
            results.append(evaluate_keras_model(
                args.keras_model,
                args.data_dir,
                args.split,
                args.model_type,
                args.max_samples,
                npu_export=True,
            ))
    for path in args.models:
        if not os.path.exists(path):
            print(f"[건너뜀] {path} 없음")
            continue
        results.append(evaluate(path, args.data_dir, args.split, args.model_type, args.max_samples))

    if len(results) > 1:
        print("\n===== float vs int8 비교 =====")
        print(f"{'model':40s} {'accuracy':>8s} {'APCER':>8s} {'BPCER':>8s} {'ACER':>8s}")
        for r in results:
            print(f"{os.path.basename(r['model']):40s} {r['accuracy']:8.4f} {r['apcer']:8.4f} {r['bpcer']:8.4f} {r['acer']:8.4f}")
    if args.report_json:
        write_regression_report(results, args.report_json, args.split)
    if args.report_csv:
        write_metrics_csv(results, args.report_csv, args.split)


if __name__ == "__main__":
    main()
