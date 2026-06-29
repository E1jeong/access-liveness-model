import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
import random
from classes import CLASS_MAPPING
from utils import (
    _sort_subject_dirs, _split_kfold_subjects,
    gather_frame_items, validate_kfold_coverage,
)

# Windows 콘솔 인코딩 설정
sys.stdout.reconfigure(encoding='utf-8')

class DualInputDataset(Dataset):
    def __init__(self, data_list, transform_rgb=None, transform_ir=None, augment=False):
        """
        data_list: list of (rgb_path, ir_path, label) tuples
        augment: 학습 시에만 True — RGB/IR 공동 공간 변환 적용
        """
        self.data_list = data_list
        self.transform_rgb = transform_rgb
        self.transform_ir = transform_ir
        self.augment = augment

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        rgb_path, ir_path, label = self.data_list[idx]

        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            raise ValueError(f"Failed to read RGB image: {rgb_path}")
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        if ir_img is None:
            raise ValueError(f"Failed to read IR image: {ir_path}")

        # RGB/IR에 동일한 공간 변환을 적용해야 두 채널이 정렬 상태를 유지한다.
        if self.augment:
            if random.random() < 0.5:
                rgb_img = cv2.flip(rgb_img, 1)
                ir_img = cv2.flip(ir_img, 1)
            angle = random.uniform(-10, 10)
            h, w = rgb_img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rgb_img = cv2.warpAffine(rgb_img, M, (w, h), flags=cv2.INTER_LINEAR)
            ir_img = cv2.warpAffine(ir_img, M, (w, h), flags=cv2.INTER_LINEAR)

        if self.transform_rgb:
            rgb_tensor = self.transform_rgb(rgb_img)
        else:
            rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0

        if self.transform_ir:
            ir_tensor = self.transform_ir(ir_img[:, :, np.newaxis])
        else:
            ir_tensor = torch.from_numpy(ir_img).unsqueeze(0).float() / 255.0

        return rgb_tensor, ir_tensor, label


def get_data_loaders(data_dir="dataset/raw", batch_size=8, k_folds=5, fold_idx=0, seed=42, num_workers=4):
    """
    학습용(Train) 및 검증용(Val) 듀얼 인풋(RGB + IR) DataLoader를 생성합니다.
    """
    train_transform_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_ir = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    if k_folds < 2:
        raise ValueError("k_folds는 2 이상이어야 합니다.")
    if fold_idx < 0 or fold_idx >= k_folds:
        raise ValueError(f"fold_idx는 0 이상 {k_folds - 1} 이하이어야 합니다.")

    train_items = []
    val_items = []

    for category, label in CLASS_MAPPING.items():
        cat_path = os.path.join(data_dir, category)
        if not os.path.exists(cat_path):
            print(f"[-] 경고: {cat_path} 디렉토리가 존재하지 않습니다.")
            continue

        subdirs = _sort_subject_dirs(cat_path, category)
        if len(subdirs) < k_folds:
            raise ValueError(
                f"{category} 클래스의 subject 폴더 수({len(subdirs)})가 K({k_folds})보다 적습니다."
            )

        train_subdirs, val_subdirs, _ = _split_kfold_subjects(subdirs, k_folds, fold_idx, seed)
        train_items.extend(gather_frame_items(cat_path, train_subdirs, label))
        val_items.extend(gather_frame_items(cat_path, val_subdirs, label))

    train_rgb_paths = {item[0] for item in train_items}
    val_rgb_paths = {item[0] for item in val_items}
    assert train_rgb_paths.isdisjoint(val_rgb_paths), "train/val rgb_path가 겹칩니다."

    train_dataset = DualInputDataset(
        train_items, transform_rgb=train_transform_rgb, transform_ir=transform_ir, augment=True
    )
    val_dataset = DualInputDataset(
        val_items, transform_rgb=val_transform_rgb, transform_ir=transform_ir, augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0
    )

    print(f"[데이터셋 구성 완료]")
    print(f" - 매핑 정보: {CLASS_MAPPING}")
    print(f" - K-fold: {k_folds}개 중 fold {fold_idx}")
    print(f" - 학습용 데이터 수: {len(train_dataset)}장 (배치 크기: {batch_size})")
    print(f" - 검증용 데이터 수: {len(val_dataset)}장")
    print(f" - DataLoader num_workers: {num_workers}, pin_memory: True")

    return train_loader, val_loader


if __name__ == "__main__":
    validate_kfold_coverage("dataset/raw", k_folds=5)
    train_loader, val_loader = get_data_loaders("dataset/raw", batch_size=4, k_folds=5, fold_idx=0)
    if len(train_loader) > 0:
        rgb_batch, ir_batch, labels = next(iter(train_loader))
        print(f"배치 RGB 텐서 크기: {rgb_batch.shape}")  # [B, 3, 224, 224]
        print(f"배치 IR 텐서 크기: {ir_batch.shape}")    # [B, 1, 224, 224]
        print(f"배치 라벨 값들: {labels}")
