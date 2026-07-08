"""학습한 tflite 모델(float/int8)을 검증셋으로 평가합니다.

INT8 양자화 후에도 보안 지표(APCER/BPCER/ACER)가 유지되는지 확인하는 용도.
입력은 학습과 동일하게 정규화한 뒤, 모델이 int8이면 scale/zero_point로 양자화하고,
출력이 int8이면 역양자화하여 float 모델과 동일 기준으로 지표를 계산합니다.
"""

import argparse
import os
import sys

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


def evaluate(model_path, data_dir, folds, fold_idx, seed, model_type, max_samples=None):
    from keras_pipeline.tf_dataset import collect_items, load_sample
    from utils import calculate_validation_metrics, quantize_for_tflite, dequantize_from_tflite
    from classes import CLASS_NAMES

    interp = _make_interpreter(model_path)
    in_details = interp.get_input_details()
    out_detail = interp.get_output_details()[0]

    def describe(d):
        shape = [int(x) for x in d['shape']]
        if len(shape) == 4 and shape[1] in (1, 3):
            return "NCHW", shape[1]
        return "NHWC", shape[-1]

    metas = [(d, *describe(d)) for d in in_details]

    rgb_d, rgb_layout = None, None
    ir_d, ir_layout = None, None

    if model_type in ("dual", "multimodal"):
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

    _, val_items = collect_items(data_dir, k_folds=folds, fold_idx=fold_idx, seed=seed)
    print(f"[데이터셋 구성 완료] K-fold: {folds}개 중 fold {fold_idx}, 검증용 데이터 수: {len(val_items)}장")

    all_labels, all_preds = [], []
    total = len(val_items)
    if max_samples is not None:
        total = min(total, max_samples)
    pbar = tqdm(total=total, desc=f"평가 {os.path.basename(model_path)}")
    for rgb_path, ir_path, label in val_items[:total]:
        rgb_hwc, ir_hwc = load_sample(rgb_path, ir_path, augment=False)
        if model_type in ("dual", "multimodal"):
            interp.set_tensor(rgb_d['index'], quantize_for_tflite(build(rgb_hwc, rgb_layout), rgb_d))
            interp.set_tensor(ir_d['index'], quantize_for_tflite(build(ir_hwc, ir_layout), ir_d))
        elif model_type == "crop_rgb":
            interp.set_tensor(rgb_d['index'], quantize_for_tflite(build(rgb_hwc, rgb_layout), rgb_d))
        elif model_type == "crop_ir":
            interp.set_tensor(ir_d['index'], quantize_for_tflite(build(ir_hwc, ir_layout), ir_d))

        interp.invoke()
        logits = dequantize_from_tflite(interp.get_tensor(out_detail['index']), out_detail)[0]
        all_labels.append(int(label))
        all_preds.append(int(np.argmax(logits)))
        pbar.update(1)
    pbar.close()

    cm, recalls, apcer, bpcer, acer = calculate_validation_metrics(all_labels, all_preds)
    acc = sum(int(l == p) for l, p in zip(all_labels, all_preds)) / len(all_labels)

    in_dtype = rgb_d['dtype'].__name__
    print(f"\n===== 평가: {model_path} (입력 dtype={in_dtype}, {len(all_labels)}장) =====")
    print(f" val_acc: {acc:.4f}")
    print(f" APCER: {apcer:.4f} | BPCER: {bpcer:.4f} | ACER: {acer:.4f}")
    print(" 클래스별 Recall:")
    for name, r in zip(CLASS_NAMES, recalls):
        print(f"   {name}: {r:.4f}")
    return {"model": model_path, "val_acc": acc, "apcer": apcer, "bpcer": bpcer, "acer": acer}


def main():
    parser = argparse.ArgumentParser(description="Evaluate float/int8 tflite on the validation set")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["model/anti_spoofing.tflite", "model/anti_spoofing_float.tflite"],
        help="평가할 tflite 경로들. 기본값은 int8과 float 중간 산출물 비교",
    )
    parser.add_argument("--data-dir", default="dataset/raw")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-type",
        choices=["dual", "multimodal", "crop_rgb", "crop_ir"],
        default="dual",
        help="평가할 모델 종류"
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    results = []
    for path in args.models:
        if not os.path.exists(path):
            print(f"[건너뜀] {path} 없음")
            continue
        results.append(evaluate(path, args.data_dir, args.folds, args.fold_idx, args.seed, args.model_type, args.max_samples))

    if len(results) > 1:
        print("\n===== float vs int8 비교 =====")
        print(f"{'model':40s} {'val_acc':>8s} {'APCER':>8s} {'BPCER':>8s} {'ACER':>8s}")
        for r in results:
            print(f"{os.path.basename(r['model']):40s} {r['val_acc']:8.4f} {r['apcer']:8.4f} {r['bpcer']:8.4f} {r['acer']:8.4f}")


if __name__ == "__main__":
    main()
