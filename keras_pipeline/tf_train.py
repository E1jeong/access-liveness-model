"""Keras 안티스푸핑 학습 진입점.

실행 경로: `scripts/keras/run_fixed_split.sh` → `scripts/keras/run_keras_train.sh`
→ `python -m keras_pipeline.tf_train`. bare `python`으로 직접 부르면
`.venv-tf`가 필요로 하는 `LD_LIBRARY_PATH`(libcudnn)가 설정되지 않아 GPU를 놓친다.

전체 흐름:
  1) 지표 방향 self-check (APCER가 뒤집혀 있으면 즉시 중단)
  2) 고정 split(train/validation/test) 누수 검증 + 파일 목록 수집
  3) tf.data 파이프라인 구성 (train은 증강+셔플, validation은 고정+캐시)
  4) MobileNetV2 기반 모델 생성 → compile → fit
  5) 학습곡선 PNG와 run metadata JSON 저장
"""
import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# GPU 메모리를 시작 시점에 전부 선점하지 않고 필요한 만큼만 늘려 잡는다.
# GTX 1660 Ti(6GB) 한 장에서 학습하므로, 다른 프로세스(변환/평가)와 동시에 떠도
# OOM으로 죽지 않게 하려는 설정. import 직후 즉, 어떤 텐서도 만들기 전에 호출해야 한다.
for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)

# `python -m keras_pipeline.tf_train`이 아니라 파일 경로로 실행되는 경우에도
# 저장소 루트의 `classes.py` / `utils.py`를 import할 수 있도록 sys.path에 루트를 넣는다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classes import CLASS_NAMES
from utils import (
    calculate_validation_metrics,
    collect_split_items,
    validate_fixed_split_coverage,
)
from keras_pipeline.tf_dataset import (
    make_dataset, make_single_dataset
)
from keras_pipeline.tf_model import (
    build_dual_mobilenetv2, build_single_mobilenetv2
)
from keras_pipeline.run_metadata import make_run_id, write_run_metadata
from keras_pipeline.artifact_paths import (
    keras_checkpoint_path,
    learning_curves_path,
    metadata_path as artifact_metadata_path,
    check_no_overwrite,
)


# 모든 스푸핑을 live로 예측한 극단 입력을 넣어 APCER가 1.0이 나오는지 학습 시작 전에 확인한다(지표 방향이 뒤집힌 채 학습하는 사고 방지).
#
# APCER = "공격(spoof)을 live로 통과시킨 비율"이므로, 정의상 모든 spoof를 live로
# 찍으면 반드시 1.0이어야 한다. 만약 여기서 0.0이 나오면 지표가 뒤집혀 있다는 뜻이고,
# 그 상태로 몇 시간 학습하면 "ACER 최저" 체크포인트가 사실은 최악의 모델이 된다.
# 학습 시작 전 1초 안에 끝나는 검사로 그 사고를 막는다.
def _run_apcer_self_check():
    labels = list(range(1, len(CLASS_NAMES)))  # live(0)를 뺀 전 스푸핑 클래스 라벨
    preds = [0] * len(labels)                  # 전부 live(0)로 예측했다고 가정
    _, _, apcer, _, _ = calculate_validation_metrics(labels, preds)
    assert apcer == 1.0, f"APCER self-check failed: {apcer}"
    print("[APCER self-check passed] all-spoof-as-live gives APCER=1.0")


# 에폭별 train loss와 train acc, 그리고 검증 (1-ACER)를 한 장의 PNG로 저장한다.
#
# history: model.fit이 반환한 History (train 지표만 들어 있다)
# val_acers: AcerCheckpoint가 에폭마다 append한 검증 ACER 리스트
# 왼쪽 그래프는 손실, 오른쪽 그래프는 정확도 계열을 겹쳐 과적합 시점을 눈으로 보게 한다.
def _save_learning_curves(history, val_acers, output_dir, model_type):
    epochs = range(1, len(history.history["loss"]) + 1)  # x축은 1부터 시작하는 에폭 번호
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["acc"], label="Train Acc")
    if val_acers:
        # ACER는 낮을수록 좋고 Acc는 높을수록 좋아 방향이 반대다. 한 축에 겹쳐 그리려고
        # 1-ACER로 뒤집어 "높을수록 좋음"으로 통일한다. 학습이 중간에 끊겨 val_acers가
        # 더 짧을 수 있으므로 x축도 그 길이만큼만 자른다.
        plt.plot(epochs[:len(val_acers)], [1 - a for a in val_acers], label="Val (1-ACER)", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Accuracy / Val ACER")
    plt.legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    # 파일명 규칙은 artifact_paths에 한 곳으로 모여 있다(dual은 접미사 없음, 나머지는 _crop_rgb 등).
    out_path = learning_curves_path(output_dir, model_type)
    plt.savefig(out_path)
    plt.close()
    print(f"[learning curves saved] {out_path}")


# 매 에폭 끝에 검증셋을 평가해, loss나 accuracy가 아니라 ACER가 최저일 때만 체크포인트를 저장하는 콜백(제품 통과 기준이 ACER이므로).
#
# 왜 기본 ModelCheckpoint(monitor="val_loss")를 쓰지 않는가:
#  - 클래스가 10개인데 spoof 9 : live 1로 심하게 불균형이라 accuracy는 live를 다 틀려도 높게 나온다.
#  - 제품 판정 기준은 APCER(공격 통과율)와 BPCER(정상 거부율)의 평균인 ACER 하나뿐이다.
#  - 따라서 "최고의 모델"은 ACER 최저 에폭이고, 그 순간에만 파일을 덮어써야 한다.
class AcerCheckpoint(tf.keras.callbacks.Callback):
    # 검증 데이터셋과 저장 경로를 받고, 최저 ACER 추적 상태를 초기화한다.
    def __init__(self, val_ds, output_path):
        super().__init__()
        self.val_ds = val_ds                 # 셔플·증강 없는 검증 데이터셋(캐시됨)
        self.output_path = output_path       # 갱신될 때마다 덮어쓸 .keras 경로
        self.best_acer = float("inf")        # 지금까지 본 최저 ACER (첫 에폭은 무조건 갱신)
        self.best_metrics = None             # 그 에폭의 acc/apcer/bpcer/acer 스냅샷
        self.acer_history = []               # 학습곡선 PNG에 그릴 에폭별 ACER
        self._val_labels = None              # 정답 라벨 캐시(아래 참고)

    # 검증셋 전체를 예측해 혼동행렬·클래스별 recall·APCER/BPCER/ACER를 출력하고, 최저 ACER가 갱신된 에폭에만 모델을 저장한다.
    def on_epoch_end(self, epoch, logs=None):
        if self._val_labels is None:
            # val_ds는 셔플 없는 캐시 데이터셋이라 순서가 고정 — 라벨은 1회만 추출
            # (매 에폭 반복하면 디코딩/전처리를 다시 돌게 되어 낭비다. 순서가 고정이라는
            #  전제가 깨지면 predict 결과와 라벨이 어긋나므로, val_ds에 shuffle을 절대 넣지 말 것.)
            self._val_labels = np.concatenate(
                [batch_labels.numpy() for _, batch_labels in self.val_ds]
            )
        # predict는 컴파일된 그래프 경로를 타므로 배치별 eager 호출보다 빠르다
        # 반환값은 softmax가 아닌 raw logits (모델 마지막 층에 활성화가 없다).
        # argmax만 쓸 것이므로 softmax를 통과시킬 필요가 없다 — 순서가 바뀌지 않는다.
        logits = self.model.predict(self.val_ds, verbose=0)
        labels = self._val_labels
        preds = np.argmax(logits, axis=1)  # 샘플별 최고 점수 클래스 인덱스

        # cm: (10,10) 혼동행렬, recalls: 클래스별 재현율, 나머지는 스푸핑 지표
        cm, recalls, apcer, bpcer, acer = calculate_validation_metrics(labels, preds)
        acc = float(np.mean(labels == preds))  # 참고용 전체 정확도(선택 기준 아님)

        print("\n -> Confusion Matrix (row=true, col=pred):")
        print(cm)
        print(" -> Class recall:")
        for class_name, recall in zip(CLASS_NAMES, recalls):
            print(f"    {class_name}: {recall:.4f}")
        print(f" -> Val Acc: {acc:.4f} | APCER: {apcer:.4f} | BPCER: {bpcer:.4f} | ACER: {acer:.4f}")

        self.acer_history.append(acer)

        # 동점(acer == best_acer)은 갱신하지 않는다 → 같은 성능이면 먼저 도달한
        # (덜 학습된 = 덜 과적합된) 에폭을 남긴다.
        if acer < self.best_acer:
            self.best_acer = acer
            self.best_metrics = {
                "val_acc": acc,
                "apcer": apcer,
                "bpcer": bpcer,
                "acer": acer,
            }
            dirpath = os.path.dirname(self.output_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            # 가중치만이 아니라 모델 전체(.keras)를 저장한다. 변환 단계
            # (convert_keras_to_tflite.py)가 구조 재생성 없이 그대로 로드해야 하기 때문.
            self.model.save(self.output_path)
            print(f" >>> Best ACER updated ({acer:.4f}) -> saved {self.output_path}")


# 학습 CLI 인자 파서. 여기 default가 곧 학습 설정의 기본값이며, 그대로 run metadata에 기록된다.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Keras anti-spoofing model with fixed train/validation/test splits."
    )
    # 데이터 루트. 하위에 train/validation/test 세 고정 split이 반드시 있어야 한다(K-Fold는 폐기됨).
    parser.add_argument("--data-dir", default="dataset/raw")
    # .keras 체크포인트·학습곡선·metadata가 모두 여기에 떨어진다. gitignore 대상.
    parser.add_argument("--output-dir", default="model/keras")
    parser.add_argument(
        "--model-type",
        choices=["dual", "crop_rgb", "crop_ir"],
        default="dual",
        help="학습할 모델 종류 (dual: 2입력, crop_rgb: 단일 RGB, crop_ir: 단일 IR)"
    )
    # 에폭 수. steps_per_epoch × epochs가 곧 CosineDecay의 총 스텝이 되므로,
    # 이 값을 바꾸면 학습률 스케줄 모양 자체가 바뀐다(단순히 더/덜 도는 게 아니다).
    parser.add_argument("--epochs", type=int, default=10)
    # 6GB VRAM에서 224x224 백본 2개(dual)를 올리기 위한 보수적 기본값.
    parser.add_argument("--batch-size", type=int, default=8)
    # CosineDecay의 시작 학습률. 사전학습 백본 미세조정이라 1e-4 수준으로 낮게 잡는다.
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    # 가중치 초기화·셔플·증강 난수를 모두 지배하는 시드(재현성).
    parser.add_argument("--seed", type=int, default=42)
    # RGB 백본 초기 가중치. "none"이면 처음부터 학습(스크래치).
    parser.add_argument("--rgb-weights", choices=["imagenet", "none"], default="imagenet")
    # 분류 헤드의 Dropout 비율. 0이면 Dropout 층 자체를 넣지 않는다.
    parser.add_argument("--dropout", type=float, default=0.2)
    # 정답 목표를 1.0이 아니라 0.91로 낮춰(=y*(1-ε)+ε/K) 과확신에 벌점을 준다.
    # 0이면 label smoothing 없이 sparse CE로 대체된다.
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing factor (default: 0.1)")
    # 백본 특징(dual이면 1280×2=2560차원)과 logits 사이 은닉층 폭. 0이면 은닉층 생략.
    parser.add_argument("--classifier-units", type=int, default=1024)
    # 지정하면 IR(1채널) 백본을 ImageNet 이식 없이 랜덤 초기화로 시작한다.
    parser.add_argument("--no-gray-imagenet-init", action="store_true")
    # IR 백본에 ImageNet Conv1(3채널)을 이식할 때 채널축을 어떻게 접을지.
    # 커널 shape (3,3,3,32)의 axis=2(입력 채널)를 sum 또는 mean으로 1로 줄인다.
    #
    #   sum  : 회색 값 g가 RGB 세 채널에 똑같이 들어왔을 때의 원본 응답과 정확히 같다.
    #          k_R·g + k_G·g + k_B·g = (k_R+k_G+k_B)·g
    #          → 사전학습 필터의 응답 크기가 보존되고, 뒤따르는 BatchNorm이 갖고 있는
    #            ImageNet 통계(이동평균/분산)와도 스케일이 맞는다.
    #   mean : 그 1/3 크기가 되어 첫 층 출력이 BatchNorm 통계보다 작아지고,
    #          초기 몇 에폭을 스케일 재적응에 낭비한다. → 이 저장소에서 기각된 변형.
    #
    # 2026-08-10부터 sum이 run_fixed_split.sh·양쪽 파서·build_* 시그니처의 기본값이라
    # 일반 실행에는 이 플래그를 넘길 필요가 없다. 지표를 공개할 실행에 mean을 넘기면
    # 안 된다(AGENTS.md의 고정 경계).
    # --no-gray-imagenet-init을 켜면 이식 자체를 하지 않으므로 이 값은 무시된다.
    parser.add_argument(
        "--conv1-reduction",
        choices=["mean", "sum"],
        default="sum",
        help="1채널 Conv1 가중치 이식 시 축소 방식 (mean: 평균, sum: 합산)"
    )
    # 산출물 파일명·metadata 키가 되는 실행 식별자.
    parser.add_argument("--run-id", help="실행 metadata에 기록할 ID (기본: UTC timestamp + model type)")
    # 기본은 덮어쓰기 금지. 이전 학습 결과를 실수로 날리는 사고를 막는다.
    parser.add_argument("--force", action="store_true", help="기존 산출물을 덮어쓰기 허용")
    return parser.parse_args()


# 학습 전체 흐름: split 누수 검증 → 데이터셋 구성 → 모델 생성 → compile → fit → 학습곡선과 run metadata 저장.
def main():
    args = parse_args()
    # python random / numpy / tensorflow 세 난수원을 한 번에 고정한다.
    # 모델 생성보다 먼저 호출해야 가중치 초기화까지 재현된다.
    # 난수가 총 4군데에 쓰이는데, 난수는 필요하지만 통제되어야 해서 학습 시작전 랜덤 난수 하나 고정.
    tf.keras.utils.set_random_seed(args.seed)

    # (1) 지표가 뒤집히지 않았는지 먼저 확인 — 실패하면 학습을 시작조차 하지 않는다.
    _run_apcer_self_check()
    # (2) split 누수 검증. subject/실경로/파일내용 해시/meta.json 세션·비디오 ID까지 교차 검사하고,
    #     하나라도 겹치면 예외를 던져 학습을 막는다. test split은 여기서 개수만 세고 건드리지 않는다.
    split_counts = validate_fixed_split_coverage(args.data_dir)
    # (3) 실제 학습/검증에 쓸 (cropRGB 경로, cropIR 경로, label) 튜플 목록.
    #     이 시점에는 경로만 들고 있고 이미지는 아직 읽지 않는다.
    train_items = collect_split_items(args.data_dir, "train")
    val_items = collect_split_items(args.data_dir, "validation")

    print("[dataset]")
    print(f" - train images: {len(train_items)}")
    print(f" - validation images: {len(val_items)}")
    print(f" - test images (isolated): {split_counts['test']}")
    print(f" - model type: {args.model_type}")

    # (4) tf.data 파이프라인 구성.
    #  - train: 셔플 O, 증강 O, .repeat()로 무한 반복 (fit이 steps_per_epoch로 끊는다)
    #  - val  : 셔플 X, 증강 X, .cache()로 첫 에폭 이후 디코딩 결과를 메모리에 재사용
    #    → val은 순서가 매 에폭 동일해야 AcerCheckpoint의 라벨 캐시와 예측이 일치한다.
    if args.model_type == "dual":
        train_ds = make_dataset(
            train_items, batch_size=args.batch_size, shuffle=True, seed=args.seed, augment=True
        ).repeat()
        val_ds = make_dataset(val_items, batch_size=args.batch_size, shuffle=False, seed=args.seed).cache()
    else:
        # crop_rgb / crop_ir는 items 튜플에서 해당 모달리티 경로 하나만 뽑아 쓴다.
        train_ds = make_single_dataset(
            train_items, input_type=args.model_type, batch_size=args.batch_size, shuffle=True, seed=args.seed, augment=True
        ).repeat()
        val_ds = make_single_dataset(val_items, input_type=args.model_type, batch_size=args.batch_size, shuffle=False, seed=args.seed).cache()

    # 무한 반복 데이터셋이므로 "1 에폭"의 길이를 직접 정해줘야 한다.
    # ceil을 써서 마지막 자투리 배치까지 한 에폭에 포함시킨다(= 전 샘플 1회 통과 분량).
    steps_per_epoch = math.ceil(len(train_items) / args.batch_size)

    # 학습률 스케줄: 코사인 곡선을 따라 initial_learning_rate → initial×alpha 로 부드럽게 감소.
    # decay_steps를 전체 학습 스텝과 같게 잡았으므로 마지막 스텝에서 정확히 바닥(1e-4×0.01=1e-6)에 닿는다.
    # alpha=0.01은 0이 아니라 아주 작은 값으로 남겨, 마지막 에폭에도 미세 조정이 계속되게 한다.
    total_steps = args.epochs * steps_per_epoch
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=total_steps,
        alpha=0.01,
    )

    # keras.applications는 "가중치 없음"을 문자열 "none"이 아니라 None으로 받는다.
    rgb_weights = None if args.rgb_weights == "none" else args.rgb_weights

    # (5) 모델 생성. 여기서 ImageNet 가중치 다운로드/이식이 일어난다.

    if args.model_type == "dual":
        model = build_dual_mobilenetv2(
            rgb_weights=rgb_weights,
            dropout=args.dropout,
            classifier_units=args.classifier_units,
            gray_imagenet_init=not args.no_gray_imagenet_init,
            conv1_reduction=args.conv1_reduction,
        )
    else:
        model = build_single_mobilenetv2(
            input_type=args.model_type,
            rgb_weights=rgb_weights,
            dropout=args.dropout,
            classifier_units=args.classifier_units,
            gray_imagenet_init=not args.no_gray_imagenet_init,
            conv1_reduction=args.conv1_reduction,
        )

    # (6) 손실 함수 선택.
    # from_logits=True: 모델 마지막 층에 softmax가 없고 raw logits을 뱉기 때문.
    #   (softmax를 모델 안에 넣지 않는 이유는 수치 안정성 + INT8 변환 시 불필요한 연산 제거)
    if args.label_smoothing > 0:
        cce_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=args.label_smoothing
        )
        # 데이터셋의 정수 라벨을 one-hot으로 바꿔 넘기는 어댑터. label_smoothing은 CategoricalCrossentropy에만 있어서 필요하다.
        # 예) label_smoothing=0.1, 클래스 10개, 정답 3 →
        #     one-hot [0,0,0,1,0,...] 이 [0.01,...,0.91,...,0.01] 로 완만해진다.
        def loss_fn(y_true, y_pred):
            # y_true는 (batch,) 또는 (batch,1)로 올 수 있어 reshape로 1차원으로 통일한다.
            y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            y_true_oh = tf.one_hot(y_true_int, depth=len(CLASS_NAMES))
            return cce_loss(y_true_oh, y_pred)
    else:
        # smoothing이 필요 없으면 one-hot 변환 없이 정수 라벨을 그대로 받는 sparse 버전을 쓴다.
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        # Adam에 스케줄 객체를 직접 넘기면 스텝마다 학습률이 자동으로 갱신된다.
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_fn,
        # 손실이 one-hot을 받더라도 metric은 정수 라벨 기준 Sparse 버전이어야 한다
        # (데이터셋이 내보내는 y_true가 정수이므로). name="acc"는 학습곡선에서 참조하는 키.
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    model.summary()  # 층 구성과 파라미터 수를 로그에 남겨 나중에 실행 기록과 대조할 수 있게 한다

    # (7) 저장 경로를 먼저 확정하고, 학습을 시작하기 전에 덮어쓰기 여부를 검사한다.
    #     몇 시간 학습한 뒤 마지막에 FileExistsError로 죽는 상황을 피하려는 순서다.
    output_path = keras_checkpoint_path(args.output_dir, args.model_type)
    check_no_overwrite(output_path, force=args.force)
    checkpoint = AcerCheckpoint(val_ds=val_ds, output_path=output_path)

    # (8) 학습 루프. validation_data를 넘기지 않는 이유는 검증을 AcerCheckpoint가
    #     직접(혼동행렬·APCER/BPCER 포함) 수행하기 때문 — 중복 평가를 피한다.
    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,  # .repeat() 데이터셋이라 필수
        epochs=args.epochs,
        callbacks=[checkpoint],
    )

    # (9) 학습곡선 저장.
    _save_learning_curves(history, checkpoint.acer_history, args.output_dir, args.model_type)

    # (10) 체크포인트가 한 번이라도 저장됐을 때만 run metadata를 남긴다.
    #      best_metrics가 None이면 저장된 .keras 파일도 없으므로 기록할 대상이 없다.
    if checkpoint.best_metrics:
        print("[best]")
        for key, value in checkpoint.best_metrics.items():
            print(f" - {key}: {value:.4f}")
        # run_id를 안 줬으면 "UTC시각_모델타입" 형태로 자동 생성 (예: 20260807T100157Z_dual)
        run_id = args.run_id or make_run_id(args.model_type)
        meta_path = artifact_metadata_path(args.output_dir, run_id)
        # CLI 인자 전체를 그대로 기록해 두면 "이 모델이 어떤 설정으로 나왔는가"를
        # 나중에 파일 하나로 재구성할 수 있다. run_id는 metadata 최상위에 따로 들어가므로 중복 제외.
        config = {
            key: value for key, value in vars(args).items()
            if key not in {"run_id"}
        }
        write_run_metadata(
            meta_path,
            run_id,
            config,
            args.data_dir,
            output_path,
            checkpoint.best_metrics,
        )
        print(f"[run metadata saved] {meta_path}")
    else:
        # 에폭을 0으로 주는 등 on_epoch_end가 한 번도 실행되지 않은 경우에만 도달한다.
        print("[-] No checkpoint was saved.")


if __name__ == "__main__":
    main()
