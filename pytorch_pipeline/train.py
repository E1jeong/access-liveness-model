import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from pytorch_pipeline.dataset import get_data_loaders, get_fixed_split_loaders
from pytorch_pipeline.model import get_anti_spoof_model
from classes import CLASS_NAMES
from utils import (
    validate_kfold_coverage, validate_fixed_split_coverage,
    calculate_validation_metrics
)

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def run_apcer_self_check():
    labels = list(range(1, len(CLASS_NAMES)))
    preds = [0] * len(labels)
    _, _, apcer, _, _ = calculate_validation_metrics(labels, preds)
    assert apcer == 1.0, f"APCER self-check failed: {apcer}"
    print("[APCER 점검 완료] spoof 샘플을 모두 live로 예측하면 APCER=1.0")


def _forward_batch(model, model_type, images_rgb, images_ir):
    if model_type in ("crop_ir", "single_ir"):
        return model(images_ir)
    elif model_type in ("crop_rgb", "single_rgb"):
        return model(images_rgb)
    elif model_type in ("dual", "dual_input"):
        return model(images_rgb, images_ir)
    raise ValueError(f"Unknown model_type: {model_type}")


def train_fixed_split(args, device, criterion):
    """
    고정 train/validation split 학습을 수행합니다.
    """
    print(f"\n[고정 split 데이터 검증 및 로딩]")
    validate_fixed_split_coverage(args.data_dir)
    train_loader, val_loader = get_fixed_split_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = get_anti_spoof_model(
        model_type=args.model_type,
        num_classes=len(CLASS_NAMES),
        conv1_reduction=args.conv1_reduction
    )
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 1e-2
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acer = float("inf")
    best_metrics = None

    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = save_dir / (args.save_name or f"best_{args.model_type}_mobilenetv3_fixed.pth")

    for epoch in range(args.epochs):
        current_lr = scheduler.get_last_lr()[0] if epoch > 0 else args.learning_rate
        print(f"\nEpoch {epoch+1}/{args.epochs}  LR={current_lr:.2e}")

        # [학습 모드]
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for images_rgb, images_ir, labels in tqdm(train_loader, desc="Training"):
            images_rgb = images_rgb.to(device)
            images_ir = images_ir.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = _forward_batch(model, args.model_type, images_rgb, images_ir)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images_rgb.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            total_train += images_rgb.size(0)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = (train_correct.double() / total_train).item()

        # [검증 모드]
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for images_rgb, images_ir, labels in tqdm(val_loader, desc="Validation"):
                images_rgb = images_rgb.to(device)
                images_ir = images_ir.to(device)
                labels = labels.to(device)

                outputs = _forward_batch(model, args.model_type, images_rgb, images_ir)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images_rgb.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                total_val += images_rgb.size(0)
                all_val_labels.extend(labels.cpu().tolist())
                all_val_preds.extend(preds.cpu().tolist())

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (val_correct.double() / total_val).item()
        confusion_matrix, recalls, apcer, bpcer, acer = calculate_validation_metrics(
            all_val_labels, all_val_preds
        )

        print(f" -> Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}%")
        print(f" -> Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%")
        print(" -> Confusion Matrix (row=true, col=pred):")
        print(confusion_matrix)
        print(" -> 클래스별 Recall:")
        for class_name, recall in zip(CLASS_NAMES, recalls):
            print(f"    {class_name}: {recall:.4f}")
        print(f" -> APCER: {apcer:.4f} | BPCER: {bpcer:.4f} | ACER: {acer:.4f}")

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        if acer < best_val_acer:
            best_val_acer = acer
            best_metrics = {
                "val_acc": epoch_val_acc,
                "apcer": apcer,
                "bpcer": bpcer,
                "acer": acer
            }
            torch.save(model.state_dict(), str(model_save_path))
            print(f" >>> 최저 검증 ACER 경신 ({best_val_acer:.4f}) -> {model_save_path} 저장 완료")

        scheduler.step()

    print(f"\n==========================================")
    print(f"★ [고정 분할 학습 완료] 최적 체크포인트: {model_save_path}")
    if best_metrics:
        print(f"  - ACER: {best_metrics['acer']:.4f}")
        print(f"  - APCER: {best_metrics['apcer']:.4f}")
        print(f"  - BPCER: {best_metrics['bpcer']:.4f}")
        print(f"  - Val Accuracy: {best_metrics['val_acc']*100:.2f}%")
    print(f"==========================================")

    return history, best_metrics


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 디바이스: {device}")
    run_apcer_self_check()

    criterion = nn.CrossEntropyLoss()

    if args.split_mode == "fixed":
        print(f"\n[고정 split 학습 모드] model_type={args.model_type}, conv1_reduction={args.conv1_reduction}")
        train_fixed_split(args, device, criterion)
    else:
        print(f"\n[K-Fold 교차검증 모드] 총 {args.folds}개 fold")
        validate_kfold_coverage(args.data_dir, k_folds=args.folds, seed=args.seed)
        # K-fold loop if explicitly requested


def parse_args():
    parser = argparse.ArgumentParser(description="Train PyTorch Anti-Spoofing Model")
    parser.add_argument("--model-type", choices=["crop_ir", "crop_rgb", "dual"], default="crop_ir", help="Model variant")
    parser.add_argument("--conv1-reduction", choices=["sum", "mean"], default="sum", help="IR Conv1 ImageNet reduction method")
    parser.add_argument("--split-mode", choices=["fixed", "kfold"], default="fixed", help="Dataset split strategy")
    parser.add_argument("--data-dir", default="dataset/raw", help="Path to raw dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (32 standard for 10-class balance)")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate (scaled for batch 32)")
    parser.add_argument("--output-dir", default="model/pytorch", help="Directory to save PyTorch checkpoints")
    parser.add_argument("--save-name", default=None, help="Custom filename for best checkpoint")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds (for K-Fold mode)")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    train_model(parse_args())
