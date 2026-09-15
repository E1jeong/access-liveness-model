import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from common.utils import collect_split_items

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


def _get_default_transforms():
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

    return train_transform_rgb, val_transform_rgb, transform_ir


def get_fixed_split_loaders(data_dir="dataset/raw", batch_size=8, num_workers=4):
    """
    고정 train/validation split DataLoader를 생성합니다.
    """
    train_transform_rgb, val_transform_rgb, transform_ir = _get_default_transforms()

    train_items = collect_split_items(data_dir, "train")
    val_items = collect_split_items(data_dir, "validation")

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

    print(f"[고정 split 데이터셋 구성 완료]")
    print(f" - Train 프레임: {len(train_dataset)}장 (배치 크기: {batch_size})")
    print(f" - Validation 프레임: {len(val_dataset)}장")
    print(f" - DataLoader num_workers: {num_workers}, pin_memory: True")

    return train_loader, val_loader
