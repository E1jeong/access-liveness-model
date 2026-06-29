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
from utils import validate_kfold_coverage, calculate_validation_metrics
from keras_pipeline.tf_dataset import collect_items, make_dataset
from keras_pipeline.tf_model import build_dual_mobilenetv2


def _run_apcer_self_check():
    labels = [1, 2, 3, 4]
    preds = [0, 0, 0, 0]
    _, _, apcer, _, _ = calculate_validation_metrics(labels, preds)
    assert apcer == 1.0, f"APCER self-check failed: {apcer}"
    print("[APCER self-check passed] all-spoof-as-live gives APCER=1.0")


def _save_learning_curves(history, val_acers, output_dir):
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
    out_path = os.path.join(output_dir, "learning_curves.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[learning curves saved] {out_path}")


class AcerCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, output_path):
        super().__init__()
        self.val_ds = val_ds
        self.output_path = output_path
        self.best_acer = float("inf")
        self.best_metrics = None
        self.acer_history = []

    def on_epoch_end(self, epoch, logs=None):
        labels = []
        preds = []
        for inputs, batch_labels in self.val_ds:
            logits = self.model(inputs, training=False)
            labels.extend(batch_labels.numpy().tolist())
            preds.extend(tf.argmax(logits, axis=1).numpy().tolist())

        cm, recalls, apcer, bpcer, acer = calculate_validation_metrics(labels, preds)
        acc = float(np.mean(np.asarray(labels) == np.asarray(preds)))

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Keras dual-input MobileNetV2 anti-spoofing model.")
    parser.add_argument("--data-dir", default="dataset/raw")
    parser.add_argument("--output-dir", default="model/keras")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rgb-weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--classifier-units", type=int, default=1024)
    parser.add_argument("--no-ir-imagenet-init", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    _run_apcer_self_check()
    validate_kfold_coverage(args.data_dir, k_folds=args.folds, seed=args.seed)
    train_items, val_items = collect_items(
        args.data_dir,
        k_folds=args.folds,
        fold_idx=args.fold_idx,
        seed=args.seed,
    )

    print("[dataset]")
    print(f" - train images: {len(train_items)}")
    print(f" - val images: {len(val_items)}")
    print(f" - fold: {args.fold_idx}/{args.folds - 1}")

    # val_ds는 AcerCheckpoint에서만 사용 — model.fit에 validation_data를 넘기지 않아
    # 에포크당 검증 forward pass가 1회만 실행된다.
    train_ds = make_dataset(
        train_items, batch_size=args.batch_size, shuffle=True, seed=args.seed, augment=True
    ).repeat()
    val_ds = make_dataset(val_items, batch_size=args.batch_size, shuffle=False, seed=args.seed)
    steps_per_epoch = math.ceil(len(train_items) / args.batch_size)

    # PyTorch CosineAnnealingLR(T_max=epochs, eta_min=lr*0.01)과 동일하게 전체 에포크에 걸쳐 감소
    total_steps = args.epochs * steps_per_epoch
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=total_steps,
        alpha=0.01,
    )

    rgb_weights = None if args.rgb_weights == "none" else args.rgb_weights
    model = build_dual_mobilenetv2(
        rgb_weights=rgb_weights,
        dropout=args.dropout,
        classifier_units=args.classifier_units,
        ir_imagenet_init=not args.no_ir_imagenet_init,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    model.summary()

    output_path = os.path.join(args.output_dir, f"best_model_fold{args.fold_idx}.keras")
    checkpoint = AcerCheckpoint(val_ds=val_ds, output_path=output_path)

    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        callbacks=[checkpoint],
    )

    _save_learning_curves(history, checkpoint.acer_history, args.output_dir)

    if checkpoint.best_metrics:
        print("[best]")
        for key, value in checkpoint.best_metrics.items():
            print(f" - {key}: {value:.4f}")
    else:
        print("[-] No checkpoint was saved.")


if __name__ == "__main__":
    main()
