import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 우리가 이전에 만든 dataset.py와 model.py에서 함수 가져오기
from dataset import get_data_loaders
from model import get_anti_spoof_model

def train_model():
    # --- 1. 설정 및 하이퍼파라미터 정의 ---
    epochs = 10           # 전체 데이터를 몇 번 학습할지 설정 (10번 반복)
    batch_size = 8       # 한 번에 학습할 이미지 개수 (기기 사양에 맞춰 설정)
    learning_rate = 1e-4  # 학습률 (가중치를 얼마나 세밀하게 업데이트할지 결정)
    device = torch.device("cpu") # GPU가 없으므로 CPU로 학습 진행

    print(f"학습 디바이스: {device}")

    # --- 2. 데이터 불러오기 ---
    train_loader, val_loader = get_data_loaders("dataset", batch_size=batch_size)

    # --- 3. 모델 정의 ---
    model = get_anti_spoof_model()
    model = model.to(device)

    # --- 4. 손실 함수(Loss Function) 및 옵티마이저(Optimizer) 설정 ---
    # 분류 문제에서 주로 사용하는 CrossEntropyLoss (교차 엔트로피 손실) 사용
    criterion = nn.CrossEntropyLoss()
    # 학습률을 적용하여 오차를 줄여나가는 Adam 옵티마이저 사용
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 시각화를 위해 학습 기록을 저장할 리스트들
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    print("\n[학습 시작] 총 10에포크 동안 학습을 진행합니다...")

    best_val_acc = 0.0

    # --- 5. 에포크 반복 (Training & Validation Loop) ---
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # [학습 모드]
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        # tqdm 라이브러리를 사용해 터미널에 실시간 진행 바 표시
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            # 옵티마이저의 기울기(경사) 초기화
            optimizer.zero_grad()

            # 1) 순전파 (Forward Pass): 모델에 이미지를 넣어 예측 결과 얻기
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 2) 역전파 (Backward Pass): 오차를 바탕으로 가중치 수정 방향(기울기) 계산
            loss.backward()

            # 3) 가중치 업데이트 (Optimizer Step)
            optimizer.step()

            # 오차 및 정확도 누적 계산
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1) # 가장 점수가 높은 클래스 인덱스 가져오기
            train_correct += torch.sum(preds == labels.data)
            total_train += images.size(0)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = (train_correct.double() / total_train).item()

        # [검증 모드] - 새로운 데이터로 모의고사 보기
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        # 검증 시에는 역전파를 하지 않으므로 기울기 계산을 비활성화(메모리 절약)
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                total_val += images.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (val_correct.double() / total_val).item()

        print(f" -> Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}%")
        print(f" -> Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%")

        # 기록 저장
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        # 최고 검증 정확도를 갱신하면 모델 파일 저장
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            # 모델 가중치를 'best_model.pth' 파일로 저장
            torch.save(model.state_dict(), "best_model.pth")
            print(f" >>> 최고 검증 정확도 경신 ({best_val_acc * 100:.2f}%) -> best_model.pth 저장 완료")

    print("\n[학습 종료] 모든 에포크가 끝났습니다.")
    print(f"최종 최고 검증 정확도: {best_val_acc * 100:.2f}%")

    # --- 6. 학습 결과 시각화 (그래프 그리기) ---
    plt.figure(figsize=(12, 5))

    # 1) Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss")
    plt.plot(range(1, epochs + 1), history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    # 2) Accuracy 그래프
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history["train_acc"], label="Train Acc")
    plt.plot(range(1, epochs + 1), history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    # 그래프 이미지 파일로 저장
    plt.tight_layout()
    plt.savefig("learning_curves.png")
    print("[시각화 완료] 학습 곡선 그래프가 'learning_curves.png'로 저장되었습니다.")

if __name__ == "__main__":
    train_model()
