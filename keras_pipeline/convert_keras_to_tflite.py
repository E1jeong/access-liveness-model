"""학습된 .keras 체크포인트를 안드로이드 배포용 .tflite로 변환한다.

실행: scripts/keras/run_keras_convert.sh (bare python 금지 — LD_LIBRARY_PATH 필요)

학습이 남긴 파일 하나에서 최대 세 가지 산출물을 만든다.
  --float     float32 그대로. 정확도 기준점이자 양자화 손실 측정의 비교 대상
  --int8      가중치·활성화를 int8로 양자화. 파일 크기 약 1/4, CPU 추론용
  --npu-int8  int8 + NPU(NNAPI)가 소화 가능한 그래프 구조로 재조립한 것

int8 계열은 "실수 범위를 정수 256단계에 어떻게 욱여넣을지"를 정해야 하는데,
그 범위는 코드로 알 수 없고 실제 입력을 흘려 봐야 안다. 그 표본이 calibration
데이터이며, 반드시 train split에서만 뽑는다(validation/test를 쓰면 누수).

각 산출물마다 .tflite + _manifest.json 한 쌍이 나온다. 안드로이드에는 둘을
함께 복사해야 한다.
"""
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
from keras_pipeline.tf_dataset import load_sample
from keras_pipeline.tf_model import _rgb_current_norm_to_mobilenet_range
from keras_pipeline.artifact_paths import (
    keras_checkpoint_path,
    tflite_path as artifact_tflite_path,
    sidecar_manifest_path,
    calibration_manifest_path,
    check_no_overwrite,
)


from keras_pipeline.export_validator import (
    _copy_nested_weights,
    _rgb_imagenet_norm_to_mobilenet_range,
    validate_npu_export_parity,
    build_npu_export_model,
    inspect_tflite,
    write_tflite_sidecar_manifest,
)


# 파일 경로에 상위 디렉터리가 있으면 산출물을 쓰기 전에 생성한다.
def _makedirs(path):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)


# 아직 파일로 쓰지 않은 변환 결과(바이트)를 메모리 인터프리터에 올려 서명을 검사한다.
# 디스크에 쓰기 '전'에 확인하므로, 규격이 깨진 산출물이 아예 파일로 남지 않는다.
def _validate_tflite_bytes(tflite_model, model_type):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    validate_tflite_model_signature(
        interpreter.get_input_details(), interpreter.get_output_details(), model_type
    )


# float32 변환. 양자화를 하지 않으므로 calibration 데이터가 필요 없고,
# Keras 모델을 양자화하지 않고 옮기므로 int8 결과를 비교할 float 기준점이 된다.
def convert_float(model, output_path, model_type):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    _validate_tflite_bytes(tflite_model, model_type)
    _makedirs(output_path)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"[float tflite saved] {output_path}")


# 캘리브레이션 이미지 한 장을 학습·평가 경로와 같은 디코딩·resize·정규화로 읽는다.
# 캘리브레이션에는 학습용 증강을 적용하지 않는다.
def _load_calibration_sample(item, model_type):
    rgb_path, ir_path, _ = item
    return load_sample(rgb_path, ir_path, augment=False)


# TFLite 변환기가 양자화 범위를 재기 위해 호출할 '표본 공급 함수'를 만들어 돌려준다.
# 변환기는 이 제너레이터를 돌려 각 텐서에 실제로 어떤 값이 흐르는지 관찰하고,
# 그 흐름을 바탕으로 각 텐서의 양자화 범위를 추정한다.
def _make_representative_dataset_gen(calibration_items, model_type, remap_rgb=False):
    """``remap_rgb=True``이면 RGB 채널을 MobileNet 입력 범위로 다시 매핑한다.

    이 옵션은 해당 변환을 수행하는 Lambda 층이 없는 NPU export 그래프에서만 쓴다.
    """
    # npu 경로에서는 보정 Lambda가 그래프에 없으므로 여기서 미리 [-1,1]로 바꿔 넣어야
    # 관찰되는 입력 분포가 실제 앱이 넣을 값과 일치한다.
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


# int8과 npu_int8이 공유하는 변환 본체. 넘겨받는 모델과 remap_rgb만 다르다.
#   convert_int8     : 학습 모델 그대로,      remap_rgb=False
#   convert_int8_npu : 재조립한 export 모델,  remap_rgb=True
def _convert_int8_core(keras_model, output_path, calibration_items, model_type, remap_rgb, log_label):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    # 양자화 최적화를 켠다.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # 활성화 범위를 재기 위한 표본. 이것이 없으면 가중치만 양자화되고 활성화는 float으로 남는다.
    converter.representative_dataset = _make_representative_dataset_gen(calibration_items, model_type, remap_rgb)
    # int8 내장 연산만 허용. 하나라도 int8 커널이 없는 연산이 있으면 여기서 변환이 실패한다
    # → float으로 조용히 흘러가는 혼합 그래프가 만들어지는 것을 막는다.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # 입출력 텐서까지 int8로 만든다(full integer quantization).
    # 앱은 float이 아니라 int8을 넣고 int8을 받게 되며, 변환 규칙은 매니페스트의
    # quantization scale/zero_point에 기록된다.
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    _validate_tflite_bytes(tflite_model, model_type)
    _makedirs(output_path)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"[{log_label} tflite saved] {output_path}")


# CPU용 INT8. 학습 모델 구조를 그대로 양자화한다.
def convert_int8(model, output_path, calibration_items, model_type):
    _convert_int8_core(model, output_path, calibration_items, model_type, remap_rgb=False, log_label="int8")


# NPU용 INT8. 양자화 '전에' 구조를 갈아끼우고, 그 결과가 원본과 같은지 먼저 증명한다.
# 순서가 중요하다 — 양자화까지 끝난 뒤에 비교하면 오차가 "구조를 잘못 옮겨서"인지
# "양자화 때문"인지 구분할 수 없다. 여기서는 float 대 float으로 비교하므로
# 차이가 나면 원인이 재조립 하나뿐이다.
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


# 변환할 모델·데이터·산출물 종류를 받는 CLI 인자 파서.
def parse_args():
    parser = argparse.ArgumentParser(description="저장된 Keras 모델을 TFLite로 변환합니다.")
    # 변환할 체크포인트. 생략하면 --output-dir와 --model-type으로 규칙에 따라 유도한다.
    # --h5-path는 .h5를 쓰던 시절의 옛 이름으로, 기존 스크립트 호환용 별칭이다.
    parser.add_argument(
        "--model-path",
        "--h5-path",
        dest="model_path",
        default=None,
    )
    parser.add_argument("--output-dir", default="model/keras")
    # calibration 표본을 뽑을 데이터 루트. train split만 사용한다.
    parser.add_argument("--data-dir", default="dataset/raw")
    parser.add_argument(
        "--model-type",
        choices=["dual", "crop_rgb", "crop_ir"],
        default="dual",
        help="변환할 모델 종류 (dual: 2입력, crop_rgb: 단일 RGB, crop_ir: 단일 IR)"
    )
    # calibration 표본 선정에 쓰는 시드. 같은 시드면 같은 표본이 뽑혀 양자화 결과가 재현된다.
    parser.add_argument("--seed", type=int, default=42)
    # 표본 개수. 많을수록 범위 추정이 안정되지만 변환 시간이 늘어난다.
    # 최소한 클래스 수(10) 이상이어야 한다 — 아래 stratified 선정이 이를 강제한다.
    parser.add_argument("--calibration-samples", type=int, default=500)
    # 세 산출물은 서로 독립이라 필요한 것만 골라 만들 수 있다(하나도 안 고르면 종료).
    parser.add_argument("--float", action="store_true", help="float TFLite 모델 생성")
    parser.add_argument("--int8", action="store_true", help="완전 정수 양자화 INT8 TFLite 모델 생성")
    parser.add_argument("--npu-int8", action="store_true", help="NNAPI/NPU 호환 완전 정수 양자화 INT8 TFLite 모델 생성")
    parser.add_argument("--force", action="store_true", help="기존 산출물을 덮어쓰기 허용")
    return parser.parse_args()


# 클래스별로 고르게 섞인 calibration 표본을 뽑는다(stratified sampling).
#
# 왜 무작위로 500장을 뽑지 않는가: 그냥 뽑으면 데이터가 많은 클래스에 쏠려서
# 특정 스푸핑 유형의 활성화 분포가 표본에 아예 없을 수 있다. 그러면 그 유형에서만
# 양자화 범위를 벗어나 값이 잘리고(clipping), 실기에서 그 공격만 못 잡는 모델이 된다.
#
# 방식: 클래스별로 나눠 담고 → 각각 섞고 → 라운드로빈으로 한 장씩 번갈아 뽑는다.
# 그래서 개수가 적은 클래스도 반드시 포함된다.
def select_stratified_calibration_items(items, max_samples, seed):
    # 클래스 수보다 적게 요구하면 애초에 모든 클래스를 담을 수 없다.
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

    # 시드 고정 난수. 클래스 순서와 각 클래스 내부 순서를 모두 섞는다.
    # 클래스 순서까지 섞는 이유: 요청 수가 클래스 수의 배수가 아니면 라운드로빈의
    # 앞자리 클래스가 한 장씩 더 뽑힐 수 있어, 순서 고정 시 특정 클래스가 계속 유리해진다.
    rng = random.Random(seed)
    labels = list(by_label)
    rng.shuffle(labels)
    for values in by_label.values():
        rng.shuffle(values)

    # 라운드로빈 선정: 클래스를 한 바퀴 돌며 한 장씩, max_samples를 채울 때까지.
    selected = []
    positions = {label: 0 for label in labels}  # 클래스별로 몇 번째까지 썼는지
    while len(selected) < max_samples:
        added = False
        for label in labels:
            # 이 클래스는 재고 소진 — 건너뛴다(개수가 적은 클래스가 먼저 바닥난다).
            if positions[label] >= len(by_label[label]):
                continue
            selected.append(by_label[label][positions[label]])
            positions[label] += 1
            added = True
            if len(selected) == max_samples:
                break
        # 한 바퀴를 돌았는데 하나도 못 담았다면 전 클래스가 소진된 것이다.
        # 요청 개수를 못 채워도 여기서 멈춘다(무한 루프 방지).
        if not added:
            break

    # 최종 확인: 라운드로빈 특성상 한 클래스도 빠질 수 없지만, 선정 로직이 바뀌어도
    # 클래스 누락이 조용히 통과하지 않도록 결과를 다시 센다.
    selected_by_label = {label: 0 for label in by_label}
    for _, _, label in selected:
        selected_by_label[label] += 1
    if any(count == 0 for count in selected_by_label.values()):
        raise ValueError("stratified calibration이 모든 클래스를 포함하지 못했습니다")

    # 두 번째 반환값은 매니페스트에 기록할 리포트다
    # (클래스별 재고 수 vs 실제 선정 수 — 나중에 쏠림 여부를 확인할 근거).
    return selected, {
        "available_by_class": {
            CLASS_NAMES[label]: len(by_label[label]) for label in by_label
        },
        "selected_by_class": {
            CLASS_NAMES[label]: selected_by_label[label] for label in selected_by_label
        },
    }


# calibration은 반드시 train split에서만 뽑는다.
# validation으로 양자화 범위를 맞추면 검증 지표가 부풀려지고, test를 쓰면
# 최종 평가가 오염된다. 이 함수가 "train"을 하드코딩해 실수 여지를 없앤다.
def collect_calibration_items(data_dir, max_samples, seed):
    """고정 train split에서만 캘리브레이션 입력을 수집한다."""
    return select_stratified_calibration_items(
        collect_split_items(data_dir, "train"), max_samples, seed
    )


# 전처리된 float32 입력 표본 한 장의 배열 크기(바이트).
# dual은 RGB 3채널 + IR 1채널 = 4채널이다. 변환기의 전체 peak 메모리는 포함하지 않는다.
def _estimated_calibration_sample_bytes(model_type):
    channels = {"dual": 4, "crop_rgb": 3, "crop_ir": 1}[model_type]
    return 224 * 224 * channels * np.dtype(np.float32).itemsize


# "이 양자화가 어떤 표본으로 이뤄졌는가"를 남기는 기록.
# 양자화 결과가 이상할 때 표본 구성부터 확인할 수 있어야 하므로, 선정된 파일 경로를
# 전부 적는다. 시드와 함께 있으면 같은 표본을 그대로 재현할 수 있다.
def write_calibration_manifest(path, items, report, model_type, seed, requested_samples):
    payload = {
        "split": "train",
        "model_type": model_type,
        "seed": seed,
        "requested_samples": requested_samples,
        "selected_samples": len(items),
        **report,
        "missing_required_samples": 0,
        # 키 이름은 기존 매니페스트 호환을 위해 유지하지만, 값은 전체 peak가 아니라
        # 전처리된 float32 입력 표본 한 장의 배열 크기다.
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
          f"sample~{payload['estimated_peak_sample_bytes']} bytes, preload=0 bytes")


# 변환 전체 흐름: 모델 로드 → 서명 검사 → (필요시) calibration 표본 수집 → 변형별 변환.
def main():
    args = parse_args()
    # 아무 변형도 고르지 않으면 할 일이 없다. 조용히 성공하지 않고 명시적으로 알린다.
    if not args.float and not args.int8 and not args.npu_int8:
        raise SystemExit("Choose at least one conversion mode: --float, --int8, and/or --npu-int8")
    # 경로를 안 줬으면 학습과 같은 규칙으로 유도한다(model/keras/best_model_fixed.keras 등).
    if args.model_path is None:
        args.model_path = keras_checkpoint_path(args.output_dir, args.model_type)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(args.model_path)

    print(f"Loading trained Keras model: {args.model_path}")
    model = tf.keras.models.load_model(
        args.model_path,
        # compile=False: 옵티마이저·손실을 복원하지 않는다. 추론만 할 것이므로 불필요하고,
        # 학습 때 쓴 커스텀 loss_fn을 되살릴 필요도 없어진다.
        compile=False,
        # 모델 안의 Lambda 층이 이 함수를 참조한다. 이름만 저장돼 있으므로
        # 로드할 때 실제 함수 객체를 알려줘야 복원된다.
        custom_objects={
            "_rgb_current_norm_to_mobilenet_range": _rgb_current_norm_to_mobilenet_range,
        },
    )
    # 검사 1회차: 변환을 시작하기 전에 입출력 규격이 계약과 맞는지 확인한다.
    validate_keras_model_signature(model, args.model_type)

    # float만 만들 때는 calibration이 필요 없으므로 이 블록 전체를 건너뛴다.
    calibration_items = None
    if args.int8 or args.npu_int8:
        # 표본을 train에서 뽑기 전에 split 누수부터 다시 확인한다.
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

    # 산출물 이름의 뿌리. "best_model_fixed.keras" → "best_model_fixed"
    # 여기에 _float / _int8 / _npu_int8 .tflite가 붙는다.
    base_name = Path(args.model_path).stem
    # 세 변형 모두 같은 4단계를 밟는다:
    #   덮어쓰기 검사 → 변환(내부에서 서명 재검사) → 텐서 정보 출력 → 매니페스트 작성
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
