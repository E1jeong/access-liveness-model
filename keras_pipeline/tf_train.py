import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)

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
def _run_apcer_self_check():
    labels = list(range(1, len(CLASS_NAMES)))
    preds = [0] * len(labels)
    _, _, apcer, _, _ = calculate_validation_metrics(labels, preds)
    assert apcer == 1.0, f"APCER self-check failed: {apcer}"
    print("[APCER self-check passed] all-spoof-as-live gives APCER=1.0")


# 에폭별 train loss와 train acc, 그리고 검증 (1-ACER)를 한 장의 PNG로 저장한다.
def _save_learning_curves(history, val_acers, output_dir, model_type):
    epochs = range(1, len(history.history["loss"]) + 1)
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
        plt.plot(epochs[:len(val_acers)], [1 - a for a in val_acers], label="Val (1-ACER)", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Accuracy / Val ACER")
    plt.legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = learning_curves_path(output_dir, model_type)
    plt.savefig(out_path)
    plt.close()
    print(f"[learning curves saved] {out_path}")


# 매 에폭 끝에 검증셋을 평가해, loss나 accuracy가 아니라 ACER가 최저일 때만 체크포인트를 저장하는 콜백(제품 통과 기준이 ACER이므로).
class AcerCheckpoint(tf.keras.callbacks.Callback):
    # 검증 데이터셋과 저장 경로를 받고, 최저 ACER 추적 상태를 초기화한다.
    def __init__(self, val_ds, output_path):
        super().__init__()
        self.val_ds = val_ds
        self.output_path = output_path
        self.best_acer = float("inf")
        self.best_metrics = None
        self.acer_history = []
        self._val_labels = None

    # 검증셋 전체를 예측해 혼동행렬·클래스별 recall·APCER/BPCER/ACER를 출력하고, 최저 ACER가 갱신된 에폭에만 모델을 저장한다.
    def on_epoch_end(self, epoch, logs=None):
        if self._val_labels is None:
            # val_ds는 셔플 없는 캐시 데이터셋이라 순서가 고정 — 라벨은 1회만 추출
            self._val_labels = np.concatenate(
                [batch_labels.numpy() for _, batch_labels in self.val_ds]
            )
        # predict는 컴파일된 그래프 경로를 타므로 배치별 eager 호출보다 빠르다
        logits = self.model.predict(self.val_ds, verbose=0)
        labels = self._val_labels
        preds = np.argmax(logits, axis=1)

        cm, recalls, apcer, bpcer, acer = calculate_validation_metrics(labels, preds)
        acc = float(np.mean(labels == preds))

        print("\n -> Confusion Matrix (row=true, col=pred):")
        print(cm)
        print(" -> Class recall:")
        for class_name, recall in zip(CLASS_NAMES, recalls):
            print(f"    {class_name}: {recall:.4f}")
        print(f" -> Val Acc: {acc:.4f} | APCER: {apcer:.4f} | BPCER: {bpcer:.4f} | ACER: {acer:.4f}")

        self.acer_history.append(acer)

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
            self.model.save(self.output_path)
            print(f" >>> Best ACER updated ({acer:.4f}) -> saved {self.output_path}")


# 학습 CLI 인자 파서. 여기 default가 곧 학습 설정의 기본값이며, 그대로 run metadata에 기록된다.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Keras anti-spoofing model with fixed train/validation/test splits."
    )
    parser.add_argument("--data-dir", default="dataset/raw")
    parser.add_argument("--output-dir", default="model/keras")
    parser.add_argument(
        "--model-type",
        choices=["dual", "crop_rgb", "crop_ir"],
        default="dual",
        help="학습할 모델 종류 (dual: 2입력, crop_rgb: 단일 RGB, crop_ir: 단일 IR)"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rgb-weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing factor (default: 0.1)")
    parser.add_argument("--classifier-units", type=int, default=1024)
    parser.add_argument("--no-gray-imagenet-init", action="store_true")
    parser.add_argument(
        "--conv1-reduction",
        choices=["mean", "sum"],
        default="sum",
        help="1채널 Conv1 가중치 이식 시 축소 방식 (mean: 평균, sum: 합산)"
    )
    parser.add_argument("--run-id", help="실행 metadata에 기록할 ID (기본: UTC timestamp + model type)")
    parser.add_argument("--force", action="store_true", help="기존 산출물을 덮어쓰기 허용")
    return parser.parse_args()


# 학습 전체 흐름: split 누수 검증 → 데이터셋 구성 → 모델 생성 → compile → fit → 학습곡선과 run metadata 저장.
def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    _run_apcer_self_check()
    split_counts = validate_fixed_split_coverage(args.data_dir)
    train_items = collect_split_items(args.data_dir, "train")
    val_items = collect_split_items(args.data_dir, "validation")

    print("[dataset]")
    print(f" - train images: {len(train_items)}")
    print(f" - validation images: {len(val_items)}")
    print(f" - test images (isolated): {split_counts['test']}")
    print(f" - model type: {args.model_type}")

    if args.model_type == "dual":
        train_ds = make_dataset(
            train_items, batch_size=args.batch_size, shuffle=True, seed=args.seed, augment=True
        ).repeat()
        val_ds = make_dataset(val_items, batch_size=args.batch_size, shuffle=False, seed=args.seed).cache()
    else:
        train_ds = make_single_dataset(
            train_items, input_type=args.model_type, batch_size=args.batch_size, shuffle=True, seed=args.seed, augment=True
        ).repeat()
        val_ds = make_single_dataset(val_items, input_type=args.model_type, batch_size=args.batch_size, shuffle=False, seed=args.seed).cache()

    steps_per_epoch = math.ceil(len(train_items) / args.batch_size)

    total_steps = args.epochs * steps_per_epoch
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=total_steps,
        alpha=0.01,
    )

    rgb_weights = None if args.rgb_weights == "none" else args.rgb_weights
    
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

    if args.label_smoothing > 0:
        cce_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=args.label_smoothing
        )
        # 데이터셋의 정수 라벨을 one-hot으로 바꿔 넘기는 어댑터. label_smoothing은 CategoricalCrossentropy에만 있어서 필요하다.
        def loss_fn(y_true, y_pred):
            y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            y_true_oh = tf.one_hot(y_true_int, depth=len(CLASS_NAMES))
            return cce_loss(y_true_oh, y_pred)
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_fn,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    model.summary()

    output_path = keras_checkpoint_path(args.output_dir, args.model_type)
    check_no_overwrite(output_path, force=args.force)
    checkpoint = AcerCheckpoint(val_ds=val_ds, output_path=output_path)

    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        callbacks=[checkpoint],
    )

    _save_learning_curves(history, checkpoint.acer_history, args.output_dir, args.model_type)

    if checkpoint.best_metrics:
        print("[best]")
        for key, value in checkpoint.best_metrics.items():
            print(f" - {key}: {value:.4f}")
        run_id = args.run_id or make_run_id(args.model_type)
        meta_path = artifact_metadata_path(args.output_dir, run_id)
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
        print("[-] No checkpoint was saved.")


if __name__ == "__main__":
    main()
