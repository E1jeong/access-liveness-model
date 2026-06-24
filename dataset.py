import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys

# Windows 콘솔 인코딩 설정
sys.stdout.reconfigure(encoding='utf-8')

class DualInputDataset(Dataset):
    def __init__(self, data_list, transform_rgb=None, transform_ir=None):
        """
        data_list: list of dicts, each containing:
          - 'rgb_path': path to cropRGB.bmp
          - 'ir_path': path to cropIR.bmp
          - 'label': integer (0, 1, 2, 3)
        """
        self.data_list = data_list
        self.transform_rgb = transform_rgb
        self.transform_ir = transform_ir

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 1. RGB 이미지 로드 (BGR -> RGB)
        rgb_img = cv2.imread(item['rgb_path'])
        if rgb_img is None:
            raise ValueError(f"Failed to read RGB image: {item['rgb_path']}")
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # 2. IR 이미지 로드 (Grayscale)
        ir_img = cv2.imread(item['ir_path'], cv2.IMREAD_GRAYSCALE)
        if ir_img is None:
            raise ValueError(f"Failed to read IR image: {item['ir_path']}")
        
        # 3. 전처리 적용
        if self.transform_rgb:
            rgb_tensor = self.transform_rgb(rgb_img)
        else:
            rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0
            
        if self.transform_ir:
            # transforms.ToTensor()는 HxW나 HxWxC 포맷을 받으므로, (H, W, 1)로 만들어줌
            ir_img_expanded = ir_img[:, :, np.newaxis]
            ir_tensor = self.transform_ir(ir_img_expanded)
        else:
            ir_tensor = torch.from_numpy(ir_img).unsqueeze(0).float() / 255.0
            
        label = item['label']
        return rgb_tensor, ir_tensor, label

def get_data_loaders(data_dir="dataset/raw", batch_size=8, split_ratio=0.8):
    """
    학습용(Train) 및 검증용(Val) 듀얼 인풋(RGB + IR) 이미지 데이터를 불러오는 DataLoader를 생성합니다.
    """
    
    # 1. 이미지 전처리(Transform) 정의
    train_transform_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_transform_ir = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5], 
            std=[0.5]
        )
    ])
    
    val_transform_ir = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5], 
            std=[0.5]
        )
    ])

    # 2. 클래스 매핑 정의
    class_mapping = {
        "live": 0,
        "print": 1,
        "display": 2,
        "picture": 3
    }
    
    train_items = []
    val_items = []
    
    # 각 클래스별로 폴더(세션)를 정렬하여 학습/검증 데이터로 분리 (데이터 누수 방지)
    for category, label in class_mapping.items():
        cat_path = os.path.join(data_dir, category)
        if not os.path.exists(cat_path):
            print(f"[-] 경고: {cat_path} 디렉토리가 존재하지 않습니다.")
            continue
            
        # 하위 숫자 폴더 목록
        subdirs = [d for d in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, d))]
        # 수치 기준으로 정렬
        try:
            subdirs = sorted(subdirs, key=lambda x: int(x))
        except ValueError:
            subdirs = sorted(subdirs)
            
        # 8:2 비율 분할
        split_idx = int(len(subdirs) * split_ratio)
        train_subdirs = subdirs[:split_idx]
        val_subdirs = subdirs[split_idx:]
        
        # 파일 수집 헬퍼 함수
        def gather_files(subdirs_list):
            gathered = []
            for sd in subdirs_list:
                sd_path = os.path.join(cat_path, sd)
                rgb_path = os.path.join(sd_path, "cropRGB.bmp")
                ir_path = os.path.join(sd_path, "cropIR.bmp")
                if os.path.exists(rgb_path) and os.path.exists(ir_path):
                    gathered.append({
                        'rgb_path': rgb_path,
                        'ir_path': ir_path,
                        'label': label
                    })
            return gathered
            
        train_items.extend(gather_files(train_subdirs))
        val_items.extend(gather_files(val_subdirs))

    # 3. 커스텀 데이터셋 생성
    train_dataset = DualInputDataset(train_items, transform_rgb=train_transform_rgb, transform_ir=train_transform_ir)
    val_dataset = DualInputDataset(val_items, transform_rgb=val_transform_rgb, transform_ir=val_transform_ir)

    # 4. DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"[데이터셋 구성 완료]")
    print(f" - 매핑 정보: {class_mapping}")
    print(f" - 학습용 데이터 수: {len(train_dataset)}장 (배치 크기: {batch_size})")
    print(f" - 검증용 데이터 수: {len(val_dataset)}장")

    return train_loader, val_loader

if __name__ == "__main__":
    # 데이터셋 구성 테스트
    train_loader, val_loader = get_data_loaders("dataset/raw", batch_size=4)
    if len(train_loader) > 0:
        rgb_batch, ir_batch, labels = next(iter(train_loader))
        print(f"배치 RGB 텐서 크기: {rgb_batch.shape}")  # [B, 3, 224, 224]
        print(f"배치 IR 텐서 크기: {ir_batch.shape}")    # [B, 1, 224, 224]
        print(f"배치 라벨 값들: {labels}")               # 예: tensor([0, 2, 1, 3])
