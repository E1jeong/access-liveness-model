"""Keras 안티스푸핑 학습 진입점.

실행 경로: `scripts/keras/run_fixed_split.sh` → `scripts/keras/run_keras_train.sh`
→ `python -m keras_pipeline.tf_train`. bare `python`으로 직접 부르면
`.venv-tf`가 필요로 하는 `LD_LIBRARY_PATH`(libcudnn)가 설정되지 않아 GPU를 놓친다.

전체 흐름:
  1) 지표 방향 self-check (APCER가 뒤집혀 있으면 즉시 중단)
  2) 고정 split(train/validation/test) 누수 검증 + 파일 목록 수집
  3) tf.data 파이프라인 구성 (train은 증강+셔플, validation은 고정+캐시)
  4) MobileNetV2 / EfficientNet-Lite0 / MobileFaceNet 기반 모델 생성
     (Multi-Task Auxiliary 3D Depth 지원: --aux-depth)
  5) compile → fit → AcerCheckpoint 저장
  6) 학습곡선 PNG와 run metadata JSON 저장
"""
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
    SUPPORTED_BACKBONES, build_dual_model, build_single_model, extract_deploy_model
)
from keras_pipeline.losses import build_binary_pad_loss, build_classification_loss
from keras_pipeline.run_metadata import make_run_id, write_run_metadata
from keras_pipeline.artifact_paths import (
    keras_checkpoint_path,
    learning_curves_path,
    metadata_path as artifact_metadata_path,
    check_no_overwrite,
)


def _run_apcer_self_check():
    labels = list(range(1, len(CLASS_NAMES)))
    preds = [0] * len(labels)
    _, _, apcer, _, _ = calculate_validation_metrics(labels, preds)
    assert apcer == 1.0, f"APCER self-check failed: {apcer}"
    print("[APCER self-check passed] all-spoof-as-live gives APCER=1.0")


def _save_learning_curves(history, val_acers, output_dir, model_type, backbone):
    epochs = range(1, len(history.history["loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    # accuracy 지표 키 찾기 (acc 또는 logits_acc)
    acc_key = "acc" if "acc" in history.history else "logits_acc"
    if acc_key in history.history:
        plt.plot(epochs, history.history[acc_key], label="Train Acc")
    if val_acers:
        plt.plot(epochs[:len(val_acers)], [1 - a for a in val_acers], label="Val (1-ACER)", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Accuracy / Val ACER")
    plt.legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = learning_curves_path(output_dir, model_type, backbone)
    plt.savefig(out_path)
    plt.close()
    print(f"[learning curves saved] {out_path}")


class CombinedHistory:
    """두 단계(워밍업 + 본 학습)의 History를 하나로 합쳐 학습 곡선을 그릴 수 있게 하는 래퍼."""
    def __init__(self, h1, h2=None):
        self.history = {}
        for key, val in h1.history.items():
            self.history[key] = list(val)
        if h2 is not None:
            for key, val in h2.history.items():
                if key in self.history:
                    self.history[key].extend(val)
                else:
                    self.history[key] = list(val)


def _merge_histories(h1, h2):
    return CombinedHistory(h1, h2)


def _build_optimizer(optimizer_name, learning_rate, weight_decay=0.01, use_ema=False, ema_momentum=0.99):
    opt_kwargs = {"learning_rate": learning_rate}
    if use_ema:
        opt_kwargs["use_ema"] = True
        opt_kwargs["ema_momentum"] = ema_momentum

    if optimizer_name == "adamw":
        if hasattr(tf.keras.optimizers, "AdamW"):
            return tf.keras.optimizers.AdamW(weight_decay=weight_decay, **opt_kwargs)
        elif hasattr(tf.keras.optimizers.experimental, "AdamW"):
            return tf.keras.optimizers.experimental.AdamW(weight_decay=weight_decay, **opt_kwargs)
        else:
            raise AttributeError("현재 TensorFlow 환경에서 AdamW를 지원하지 않습니다.")
    elif optimizer_name == "adam":
        return tf.keras.optimizers.Adam(**opt_kwargs)
    else:
        raise ValueError(f"지원하지 않는 옵티마이저: {optimizer_name}")


def _set_backbone_trainable(model, trainable=True):
    count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or any(
            b in layer.name for b in ("mobilenetv2", "efficientnet", "mobilefacenet")
        ):
            layer.trainable = trainable
            count += 1
    state_str = "동결 해제(unfrozen)" if trainable else "동결(frozen)"
    print(f"[{state_str}] {count}개 백본 레이어의 trainable을 {trainable}로 설정했습니다 (학습 가능 파라미터 텐서 수: {len(model.trainable_variables)})")


class AcerCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, output_path):
        super().__init__()
        self.val_ds = val_ds
        self.output_path = output_path
        self.best_acer = float("inf")
        self.best_metrics = None
        self.acer_history = []
        self._val_labels = None

    def on_epoch_end(self, epoch, logs=None):
        if self._val_labels is None:
            self._val_labels = np.concatenate(
                [batch_labels.numpy() if not isinstance(batch_labels, dict) else batch_labels["logits"].numpy()
                 for _, batch_labels in self.val_ds]
            )

        is_ema = getattr(self.model.optimizer, "use_ema", False)
        orig_weights = None
        if is_ema:
            orig_weights = [v.numpy() for v in self.model.trainable_variables]
            self.model.optimizer.finalize_variable_values(self.model.trainable_variables)

        try:
            raw_preds = self.model.predict(self.val_ds, verbose=0)
            if isinstance(raw_preds, (list, tuple)):
                logits = raw_preds[0]
            elif isinstance(raw_preds, dict):
                logits = raw_preds["logits"]
            else:
                logits = raw_preds

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
        finally:
            if is_ema and orig_weights is not None:
                for v, orig in zip(self.model.trainable_variables, orig_weights):
                    v.assign(orig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="고정 train/validation/test split으로 Keras 안티스푸핑 모델을 학습합니다."
    )
    parser.add_argument("--data-dir", default="dataset/raw")
    parser.add_argument("--output-dir", default="model/keras")
    parser.add_argument(
        "--model-type",
        choices=["dual", "crop_rgb", "crop_ir"],
        default="dual",
        help="학습할 모델 종류 (dual: 2입력, crop_rgb: 단일 RGB, crop_ir: 단일 IR)"
    )
    parser.add_argument(
        "--backbone",
        choices=SUPPORTED_BACKBONES,
        default="mobilenetv2",
        help="특징 추출 백본 (mobilenetv2, efficientnet_lite0, mobilefacenet)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rgb-weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="라벨 스무딩 계수(기본값: 0.1)")
    parser.add_argument("--classifier-units", type=int, default=1024)
    parser.add_argument("--no-gray-imagenet-init", action="store_true")
    parser.add_argument(
        "--conv1-reduction",
        choices=["mean", "sum"],
        default="sum",
        help="1채널 Conv1 가중치 이식 시 축소 방식 (mean: 평균, sum: 합산)"
    )
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "adam"],
        default="adamw",
        help="학습 옵티마이저 (adamw, adam, 기본값: adamw)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW 가중치 감쇄율 (기본값: 0.01)",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="EMA 가중치 추적 및 체크포인트 평가/저장 활성화",
    )
    parser.add_argument(
        "--ema-momentum",
        type=float,
        default=0.99,
        help="EMA 모멘텀 계수 (기본값: 0.99)",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="초기 백본 동결 워밍업 에포크 수 (기본값: 0, 0이면 비활성화)",
    )
    parser.add_argument(
        "--aux-depth",
        action="store_true",
        help="Multi-Task 3D Depth 보조 지도학습 활성화",
    )
    parser.add_argument(
        "--depth-loss-weight",
        type=float,
        default=0.5,
        help="3D Depth 보조 손실 가중치 (기본값: 0.5)",
    )
    parser.add_argument(
        "--aux-binary-pad",
        action="store_true",
        help="Phase 2 bona-fide/spoof 보조 지도학습 활성화",
    )
    parser.add_argument(
        "--binary-pad-loss-weight",
        type=float,
        default=0.2,
        help="binary PAD 보조 손실 가중치 (기본값: 0.2)",
    )
    parser.add_argument(
        "--loss-type",
        "--loss",
        dest="loss_type",
        choices=["ce", "focal"],
        default="ce",
        help="분류 손실 함수 (ce: CrossEntropy, focal: FocalLoss, 기본값: ce)",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal Loss 감마 계수 (난이도 집중 파라미터, 기본값: 2.0)",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.25,
        help="Focal Loss 알파 계수 (클래스 밸런싱 파라미터, 기본값: 0.25)",
    )
    parser.add_argument("--run-id", help="실행 메타데이터에 기록할 ID(기본값: UTC 시각 + 모델 종류)")
    parser.add_argument("--force", action="store_true", help="기존 산출물을 덮어쓰기 허용")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.freeze_backbone_epochs < 0:
        raise ValueError("--freeze-backbone-epochs는 0 이상이어야 합니다.")
    if args.freeze_backbone_epochs >= args.epochs:
        raise ValueError(
            f"--freeze-backbone-epochs ({args.freeze_backbone_epochs})는 총 에포크({args.epochs})보다 작아야 합니다."
        )
    if args.binary_pad_loss_weight < 0:
        raise ValueError("--binary-pad-loss-weight는 0 이상이어야 합니다.")

    if args.backbone == "mobilefacenet":
        if args.model_type != "crop_ir":
            raise ValueError("MobileFaceNet은 crop_ir 단일 입력만 지원합니다")
        if args.rgb_weights != "none":
            raise ValueError("MobileFaceNet은 scratch 학습만 지원하므로 --rgb-weights none을 사용해야 합니다")

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
    print(f" - backbone: {args.backbone}")
    print(f" - optimizer: {args.optimizer} (weight_decay={args.weight_decay})")
    print(f" - use_ema: {args.use_ema} (momentum={args.ema_momentum})")
    print(f" - freeze_backbone_epochs: {args.freeze_backbone_epochs}")
    print(f" - loss_type: {args.loss_type} (gamma={args.focal_gamma}, alpha={args.focal_alpha}, label_smoothing={args.label_smoothing})")
    print(f" - aux_depth: {args.aux_depth} (depth_loss_weight={args.depth_loss_weight})")
    print(f" - aux_binary_pad: {args.aux_binary_pad} (binary_pad_loss_weight={args.binary_pad_loss_weight})")

    if args.model_type == "dual":
        train_ds = make_dataset(
            train_items, batch_size=args.batch_size, shuffle=True, seed=args.seed,
            augment=True, repeat=True, aux_depth=args.aux_depth,
            aux_binary_pad=args.aux_binary_pad
        )
        val_ds = make_dataset(val_items, batch_size=args.batch_size, shuffle=False, seed=args.seed).cache()
    else:
        train_ds = make_single_dataset(
            train_items, input_type=args.model_type, batch_size=args.batch_size, shuffle=True, seed=args.seed,
            augment=True, repeat=True, aux_depth=args.aux_depth,
            aux_binary_pad=args.aux_binary_pad
        )
        val_ds = make_single_dataset(val_items, input_type=args.model_type, batch_size=args.batch_size, shuffle=False, seed=args.seed).cache()

    steps_per_epoch = math.ceil(len(train_items) / args.batch_size)
    rgb_weights = None if args.rgb_weights == "none" else args.rgb_weights

    if args.model_type == "dual":
        model = build_dual_model(
            rgb_weights=rgb_weights,
            dropout=args.dropout,
            classifier_units=args.classifier_units,
            gray_imagenet_init=not args.no_gray_imagenet_init,
            conv1_reduction=args.conv1_reduction,
            backbone=args.backbone,
            aux_depth=args.aux_depth,
            aux_binary_pad=args.aux_binary_pad,
        )
    else:
        model = build_single_model(
            input_type=args.model_type,
            rgb_weights=rgb_weights,
            dropout=args.dropout,
            classifier_units=args.classifier_units,
            gray_imagenet_init=not args.no_gray_imagenet_init,
            conv1_reduction=args.conv1_reduction,
            backbone=args.backbone,
            aux_depth=args.aux_depth,
            aux_binary_pad=args.aux_binary_pad,
        )

    # 손실 함수 구성
    cls_loss_fn = build_classification_loss(
        loss_type=args.loss_type,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
    )

    if args.aux_depth or args.aux_binary_pad:
        loss_dict = {"logits": cls_loss_fn}
        loss_weights_dict = {"logits": 1.0}
        if args.aux_depth:
            loss_dict["depth_output"] = tf.keras.losses.MeanSquaredError()
            loss_weights_dict["depth_output"] = args.depth_loss_weight
        if args.aux_binary_pad:
            loss_dict["pad_output"] = build_binary_pad_loss()
            loss_weights_dict["pad_output"] = args.binary_pad_loss_weight
        metrics_dict = {"logits": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]}
    else:
        loss_dict = cls_loss_fn
        loss_weights_dict = None
        metrics_dict = [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]

    output_path = keras_checkpoint_path(args.output_dir, args.model_type, args.backbone)
    check_no_overwrite(output_path, force=args.force)
    checkpoint = AcerCheckpoint(val_ds=val_ds, output_path=output_path)

    if args.freeze_backbone_epochs > 0:
        print(f"\n========================================================")
        print(f" [Stage 1/2: Backbone Freeze Warmup] ({args.freeze_backbone_epochs} epoch(s))")
        print(f"========================================================")
        _set_backbone_trainable(model, trainable=False)
        warmup_optimizer = _build_optimizer(
            optimizer_name=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            use_ema=args.use_ema,
            ema_momentum=args.ema_momentum,
        )
        model.compile(
            optimizer=warmup_optimizer,
            loss=loss_dict,
            loss_weights=loss_weights_dict,
            metrics=metrics_dict,
        )
        model.summary()
        history_warmup = model.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            epochs=args.freeze_backbone_epochs,
            callbacks=[checkpoint],
        )

        remaining_epochs = args.epochs - args.freeze_backbone_epochs
        remaining_steps = remaining_epochs * steps_per_epoch
        print(f"\n========================================================")
        print(f" [Stage 2/2: Full Fine-tuning] ({remaining_epochs} epoch(s), Epochs {args.freeze_backbone_epochs + 1}~{args.epochs})")
        print(f"========================================================")
        _set_backbone_trainable(model, trainable=True)
        lr_schedule_finetune = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=remaining_steps,
            alpha=0.01,
        )
        finetune_optimizer = _build_optimizer(
            optimizer_name=args.optimizer,
            learning_rate=lr_schedule_finetune,
            weight_decay=args.weight_decay,
            use_ema=args.use_ema,
            ema_momentum=args.ema_momentum,
        )
        model.compile(
            optimizer=finetune_optimizer,
            loss=loss_dict,
            loss_weights=loss_weights_dict,
            metrics=metrics_dict,
        )
        model.summary()
        history_finetune = model.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            initial_epoch=args.freeze_backbone_epochs,
            epochs=args.epochs,
            callbacks=[checkpoint],
        )
        history = _merge_histories(history_warmup, history_finetune)
    else:
        total_steps = args.epochs * steps_per_epoch
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=total_steps,
            alpha=0.01,
        )
        optimizer = _build_optimizer(
            optimizer_name=args.optimizer,
            learning_rate=lr_schedule,
            weight_decay=args.weight_decay,
            use_ema=args.use_ema,
            ema_momentum=args.ema_momentum,
        )
        model.compile(
            optimizer=optimizer,
            loss=loss_dict,
            loss_weights=loss_weights_dict,
            metrics=metrics_dict,
        )
        model.summary()
        history = model.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            callbacks=[checkpoint],
        )

    _save_learning_curves(history, checkpoint.acer_history, args.output_dir, args.model_type, args.backbone)

    if checkpoint.best_metrics:
        print("[best]")
        for key, value in checkpoint.best_metrics.items():
            print(f" - {key}: {value:.4f}")
        run_id = args.run_id or make_run_id(args.model_type, args.backbone)
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
