import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classes import CLASS_NAMES
from keras_pipeline.tf_dataset import collect_items, make_dataset, validate_kfold_coverage
from keras_pipeline.tf_model import build_dual_mobilenetv2


def calculate_validation_metrics(labels, preds):
    num_classes = len(CLASS_NAMES)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, pred in zip(labels, preds):
        confusion_matrix[int(label), int(pred)] += 1

    recalls = []
    for class_idx in range(num_classes):
        total = confusion_matrix[class_idx, :].sum()
        correct = confusion_matrix[class_idx, class_idx]
        recalls.append(float(correct / total) if total > 0 else 0.0)

    labels = np.asarray(labels)
    preds = np.asarray(preds)
    live_mask = labels == 0
    spoof_mask = labels != 0

    total_live = int(live_mask.sum())
    total_spoof = int(spoof_mask.sum())
    apcer_errors = int(((preds == 0) & spoof_mask).sum())
    bpcer_errors = int(((preds != 0) & live_mask).sum())

    apcer = apcer_errors / total_spoof if total_spoof > 0 else 0.0
    bpcer = bpcer_errors / total_live if total_live > 0 else 0.0
    acer = (apcer + bpcer) / 2.0
    return confusion_matrix, recalls, apcer, bpcer, acer


class AcerCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, output_path):
        super().__init__()
        self.val_ds = val_ds
        self.output_path = output_path
        self.best_acer = float("inf")
        self.best_metrics = None

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

        if acer < self.best_acer:
            self.best_acer = acer
            self.best_metrics = {
                "val_acc": acc,
                "apcer": apcer,
                "bpcer": bpcer,
                "acer": acer,
            }
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
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
    return parser.parse_args()


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

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

    train_ds = make_dataset(train_items, batch_size=args.batch_size, shuffle=True, seed=args.seed).repeat()
    val_ds = make_dataset(val_items, batch_size=args.batch_size, shuffle=False, seed=args.seed)
    steps_per_epoch = math.ceil(len(train_items) / args.batch_size)
    validation_steps = math.ceil(len(val_items) / args.batch_size)

    rgb_weights = None if args.rgb_weights == "none" else args.rgb_weights
    model = build_dual_mobilenetv2(rgb_weights=rgb_weights, dropout=args.dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    model.summary()

    output_path = os.path.join(args.output_dir, f"best_model_fold{args.fold_idx}.keras")
    checkpoint = AcerCheckpoint(val_ds=val_ds, output_path=output_path)

    model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=args.epochs,
        callbacks=[checkpoint],
    )

    if checkpoint.best_metrics:
        print("[best]")
        for key, value in checkpoint.best_metrics.items():
            print(f" - {key}: {value:.4f}")
    else:
        print("[-] No checkpoint was saved.")


if __name__ == "__main__":
    main()
