import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 우리가 이전에 만든 dataset.py와 model.py에서 함수 가져오기
from dataset import get_data_loaders, validate_kfold_coverage
from model import get_anti_spoof_model
from classes import CLASS_NAMES


def calculate_validation_metrics(labels, preds):
    num_classes = len(CLASS_NAMES)
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for label, pred in zip(labels, preds):
        confusion_matrix[int(label), int(pred)] += 1

    recalls = []
    for class_idx in range(num_classes):
        total = confusion_matrix[class_idx, :].sum().item()
        correct = confusion_matrix[class_idx, class_idx].item()
        recalls.append(correct / total if total > 0 else 0.0)

    labels_tensor = torch.tensor(labels)
    preds_tensor = torch.tensor(preds)

    live_mask = labels_tensor == 0
    spoof_mask = labels_tensor != 0
    total_live = live_mask.sum().item()
    total_spoof = spoof_mask.sum().item()

    apcer_errors = ((preds_tensor == 0) & spoof_mask).sum().item()
    bpcer_errors = ((preds_tensor != 0) & live_mask).sum().item()

    apcer = apcer_errors / total_spoof if total_spoof > 0 else 0.0
    bpcer = bpcer_errors / total_live if total_live > 0 else 0.0
    acer = (apcer + bpcer) / 2

    return confusion_matrix, recalls, apcer, bpcer, acer


def run_apcer_self_check():
    labels = [1, 2, 3, 4]
    preds = [0, 0, 0, 0]
    _, _, apcer, _, _ = calculate_validation_metrics(labels, preds)
    assert apcer == 1.0, f"APCER self-check failed: {apcer}"
    print("[APCER 점검 완료] spoof 샘플을 모두 live로 예측하면 APCER=1.0")


def train_one_fold(fold_idx, args, device, criterion):
    # --- 2. 데이터 불러오기 ---
    train_loader, val_loader = get_data_loaders(
        "dataset/raw",
        batch_size=args.batch_size,
        k_folds=args.folds,
        fold_idx=fold_idx,
        seed=args.seed
    )

    # --- 3. 모델 정의 ---
    model = get_anti_spoof_model()
    model = model.to(device)

    # --- 4. 옵티마이저(Optimizer) 설정 ---
    # 학습률을 적용하여 오차를 줄여나가는 Adam 옵티마이저 사용
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 시각화를 위해 학습 기록을 저장할 리스트들
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    best_val_acer = float("inf")
    best_fold_metrics = None

    # --- 5. 에포크 반복 (Training & Validation Loop) ---
    for epoch in range(args.epochs):
        print(f"\n[Fold {fold_idx}] Epoch {epoch+1}/{args.epochs}")
        
        # [학습 모드]
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        # tqdm 라이브러리를 사용해 터미널에 실시간 진행 바 표시
        for images_rgb, images_ir, labels in tqdm(train_loader, desc="Training"):
            images_rgb, images_ir, labels = images_rgb.to(device), images_ir.to(device), labels.to(device)

            # 옵티마이저의 기울기(경사) 초기화
            optimizer.zero_grad()

            # 1) 순전파 (Forward Pass): 모델에 이미지를 넣어 예측 결과 얻기
            outputs = model(images_rgb, images_ir)
            loss = criterion(outputs, labels)

            # 2) 역전파 (Backward Pass): 오차를 바탕으로 가중치 수정 방향(기울기) 계산
            loss.backward()

            # 3) 가중치 업데이트 (Optimizer Step)
            optimizer.step()

            # 오차 및 정확도 누적 계산
            train_loss += loss.item() * images_rgb.size(0)
            _, preds = torch.max(outputs, 1) # 가장 점수가 높은 클래스 인덱스 가져오기
            train_correct += torch.sum(preds == labels.data)
            total_train += images_rgb.size(0)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = (train_correct.double() / total_train).item()

        # [검증 모드] - 새로운 데이터로 모의고사 보기
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        all_val_labels = []
        all_val_preds = []

        # 검증 시에는 역전파를 하지 않으므로 기울기 계산을 비활성화(메모리 절약)
        with torch.no_grad():
            for images_rgb, images_ir, labels in tqdm(val_loader, desc="Validation"):
                images_rgb, images_ir, labels = images_rgb.to(device), images_ir.to(device), labels.to(device)
                
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
        confusion_matrix, recalls, apcer, bpcer, acer = calculate_validation_metrics(all_val_labels, all_val_preds)

        print(f" -> Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}%")
        print(f" -> Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%")
        print(" -> Confusion Matrix (row=true, col=pred):")
        print(confusion_matrix)
        print(" -> 클래스별 Recall:")
        for class_name, recall in zip(CLASS_NAMES, recalls):
            print(f"    {class_name}: {recall:.4f}")
        print(f" -> APCER: {apcer:.4f} | BPCER: {bpcer:.4f} | ACER: {acer:.4f}")

        # 기록 저장
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        # 최저 ACER를 갱신하면 모델 파일 저장
        if acer < best_val_acer:
            best_val_acer = acer
            best_fold_metrics = {
                "val_acc": epoch_val_acc,
                "apcer": apcer,
                "bpcer": bpcer,
                "acer": acer
            }
            # 모델 가중치를 fold별 파일로 저장
            os.makedirs("model", exist_ok=True)
            model_path = f"model/best_model_fold{fold_idx}.pth"
            torch.save(model.state_dict(), model_path)
            print(f" >>> 최저 검증 ACER 경신 ({best_val_acer:.4f}) -> {model_path} 저장 완료")

    return history, best_fold_metrics


def save_learning_curves(all_histories):
    plt.figure(figsize=(12, 5))

    # 1) Loss 그래프
    plt.subplot(1, 2, 1)
    for fold_idx, history in enumerate(all_histories):
        plt.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label=f"Fold {fold_idx} Train Loss")
        plt.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], label=f"Fold {fold_idx} Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    # 2) Accuracy 그래프
    plt.subplot(1, 2, 2)
    for fold_idx, history in enumerate(all_histories):
        plt.plot(range(1, len(history["train_acc"]) + 1), history["train_acc"], label=f"Fold {fold_idx} Train Acc")
        plt.plot(range(1, len(history["val_acc"]) + 1), history["val_acc"], label=f"Fold {fold_idx} Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    # 그래프 이미지 파일로 저장
    plt.tight_layout()
    os.makedirs("model", exist_ok=True)
    plt.savefig("model/learning_curves.png")
    print("[시각화 완료] 학습 곡선 그래프가 'model/learning_curves.png'로 저장되었습니다.")


def train_model(args):
    # --- 1. 설정 및 하이퍼파라미터 정의 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU가 있으면 자동으로 사용

    print(f"학습 디바이스: {device}")
    print("단일 분할의 지표 절대값보다 K-fold 평균±표준편차를 신뢰 기준으로 봅니다.")
    run_apcer_self_check()
    validate_kfold_coverage("dataset/raw", k_folds=args.folds, seed=args.seed)

    # 분류 문제에서 주로 사용하는 CrossEntropyLoss (교차 엔트로피 손실) 사용
    criterion = nn.CrossEntropyLoss()

    print(f"\n[학습 시작] 총 {args.folds}개 fold, fold당 {args.epochs}에포크 동안 학습을 진행합니다...")

    all_histories = []
    fold_metrics = []
    total_folds = args.max_folds if args.max_folds is not None else args.folds

    for fold_idx in range(total_folds):
        print(f"\n========== Fold {fold_idx}/{args.folds - 1} ==========")
        history, best_metrics = train_one_fold(fold_idx, args, device, criterion)
        all_histories.append(history)
        if best_metrics is not None:
            fold_metrics.append(best_metrics)

    print("\n[학습 종료] 모든 요청 fold가 끝났습니다.")
    if fold_metrics:
        for metric_name in ["val_acc", "apcer", "bpcer", "acer"]:
            values = torch.tensor([m[metric_name] for m in fold_metrics], dtype=torch.float32)
            std = values.std(unbiased=False).item()
            print(f"{metric_name}: 평균 {values.mean().item():.4f} ± 표준편차 {std:.4f}")

    # --- 6. 학습 결과 시각화 (그래프 그리기) ---
    save_learning_curves(all_histories)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train_model(parse_args())
