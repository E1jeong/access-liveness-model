import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=8):
    """
    학습용(Train) 및 검증용(Val) 이미지 데이터를 불러오는 DataLoader를 생성합니다.
    """
    
    # 1. 이미지 전처리(Transform) 정의
    # AI 모델(MobileNet)은 특정 크기(예: 224x224)의 이미지와 표준화된 수치(정규화)를 입력으로 받습니다.
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),      # 이미지를 224x224 크기로 통일
        transforms.RandomHorizontalFlip(),  # 학습 데이터 증강: 좌우 반전을 무작위로 적용 (다양성 확보)
        transforms.ToTensor(),              # 이미지를 PyTorch 텐서(0~1 사이의 숫자 배열)로 변환
        transforms.Normalize(               # 이미지 채널별 평균과 표준편차를 사용하여 정규화
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),      # 검증 데이터는 무작위 변형 없이 크기 조절만 수행
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 2. ImageFolder를 사용한 데이터셋 생성
    # ImageFolder는 폴더명(real, spoof)을 기준으로 이미지와 라벨(클래스)을 자동으로 매핑합니다.
    # 예: real 폴더 안의 이미지 -> 라벨 0, spoof 폴더 안의 이미지 -> 라벨 1
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=val_transform
    )

    # 3. DataLoader 생성
    # DataLoader는 데이터셋을 배치(Batch) 단위로 나누어 모델에 공급하고, 데이터를 섞는(Shuffle) 역할을 합니다.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # 학습용 데이터는 순서를 무작위로 섞어서 학습 효과를 높임
        num_workers=0       # Windows 환경의 안정성을 위해 멀티프로세싱 워커 수를 0으로 설정
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,      # 검증 데이터는 순서를 섞을 필요가 없음
        num_workers=0
    )

    # 라벨 정보 출력 (어떤 폴더가 어떤 번호로 매핑되었는지 확인)
    # train_dataset.class_to_idx는 {'real': 0, 'spoof': 1} 형태를 가집니다.
    print(f"[데이터셋 구성 완료]")
    print(f" - 매핑 정보: {train_dataset.class_to_idx}")
    print(f" - 학습용 데이터 수: {len(train_dataset)}장 (배치 크기: {batch_size})")
    print(f" - 검증용 데이터 수: {len(val_dataset)}장")

    return train_loader, val_loader

# 독립적으로 실행 시 데이터 로드가 잘 되는지 테스트하는 코드
if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders("dataset", batch_size=4)
    # 첫 번째 배치를 가져와서 크기 확인
    images, labels = next(iter(train_loader))
    print(f"배치 이미지 텐서 크기: {images.shape}")  # [배치크기, 채널(RGB), 높이, 너비] -> [4, 3, 224, 224]
    print(f"배치 라벨 값들: {labels}")               # 예: tensor([0, 1, 0, 1])
