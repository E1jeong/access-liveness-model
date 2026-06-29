import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from dataset import get_data_loaders
from model import get_anti_spoof_model
from classes import CLASS_NAMES
from utils import validate_kfold_coverage, calculate_validation_metrics


def run_apcer_self_check():
    labels = [1, 2, 3, 4]
    preds = [0, 0, 0, 0]
    _, _, apcer, _, _ = calculate_validation_metrics(labels, preds)
    assert apcer == 1.0, f"APCER self-check failed: {apcer}"
    print("[APCER 점검 완료] spoof 샘플을 모두 live로 예측하면 APCER=1.0")


def train_one_fold(fold_idx, args, device, criterion):
    train_loader, val_loader = get_data_loaders(
        "dataset/raw",
        batch_size=args.batch_size,
        k_folds=args.folds,
        fold_idx=fold_idx,
        seed=args.seed,
        num_workers=args.num_workers
    )

    model = get_anti_spoof_model()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # 에포크가 진행될수록 학습률을 cos 곡선으로 부드럽게 감소시킨다.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 1e-2
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    best_val_acer = float("inf")
    best_fold_metrics = None

    for epoch in range(args.epochs):
        current_lr = scheduler.get_last_lr()[0] if epoch > 0 else args.learning_rate
        print(f"\n[Fold {fold_idx}] Epoch {epoch+1}/{args.epochs}  LR={current_lr:.2e}")

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
            outputs = model(images_rgb, images_ir)
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

                outputs = model(images_rgb, images_ir)
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
            best_fold_metrics = {
                "val_acc": epoch_val_acc,
                "apcer": apcer,
                "bpcer": bpcer,
                "acer": acer
            }
            os.makedirs("model", exist_ok=True)
            model_path = f"model/best_model_fold{fold_idx}.pth"
            torch.save(model.state_dict(), model_path)
            print(f" >>> 최저 검증 ACER 경신 ({best_val_acer:.4f}) -> {model_path} 저장 완료")

        scheduler.step()

    return history, best_fold_metrics


def save_learning_curves(all_histories):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for fold_idx, history in enumerate(all_histories):
        plt.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label=f"Fold {fold_idx} Train")
        plt.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], label=f"Fold {fold_idx} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    for fold_idx, history in enumerate(all_histories):
        plt.plot(range(1, len(history["train_acc"]) + 1), history["train_acc"], label=f"Fold {fold_idx} Train")
        plt.plot(range(1, len(history["val_acc"]) + 1), history["val_acc"], label=f"Fold {fold_idx} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    os.makedirs("model", exist_ok=True)
    plt.savefig("model/learning_curves.png")
    print("[시각화 완료] 학습 곡선 그래프가 'model/learning_curves.png'로 저장되었습니다.")


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 디바이스: {device}")
    print("단일 분할의 지표 절대값보다 K-fold 평균±표준편차를 신뢰 기준으로 봅니다.")
    run_apcer_self_check()
    validate_kfold_coverage("dataset/raw", k_folds=args.folds, seed=args.seed)

    criterion = nn.CrossEntropyLoss()
    total_folds = args.max_folds if args.max_folds is not None else args.folds
    print(f"\n[학습 시작] 총 {args.folds}개 fold, fold당 {args.epochs}에포크 동안 학습을 진행합니다...")

    all_histories = []
    fold_metrics = []

    for fold_idx in range(total_folds):
        print(f"\n========== Fold {fold_idx}/{args.folds - 1} ==========")
        history, best_metrics = train_one_fold(fold_idx, args, device, criterion)
        all_histories.append(history)
        if best_metrics is not None:
            fold_metrics.append(best_metrics)

    print("\n[학습 종료] 모든 요청 fold가 끝났습니다.")
    if fold_metrics:
        import numpy as np
        for metric_name in ["val_acc", "apcer", "bpcer", "acer"]:
            values = np.array([m[metric_name] for m in fold_metrics], dtype=np.float32)
            print(f"{metric_name}: 평균 {values.mean():.4f} ± 표준편차 {values.std():.4f}")

    save_learning_curves(all_histories)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    train_model(parse_args())
