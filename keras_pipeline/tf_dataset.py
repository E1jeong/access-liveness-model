import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classes import CLASS_MAPPING
from utils import (
    _sort_subject_dirs, _split_kfold_subjects,
    gather_frame_items, validate_kfold_coverage,
)

IMAGE_SIZE = (224, 224)
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IR_MEAN = np.array([0.5], dtype=np.float32)
IR_STD = np.array([0.5], dtype=np.float32)


def collect_items(data_dir="dataset/raw", k_folds=5, fold_idx=0, seed=42):
    if k_folds < 2:
        raise ValueError("k_folds must be at least 2.")
    if fold_idx < 0 or fold_idx >= k_folds:
        raise ValueError(f"fold_idx must be between 0 and {k_folds - 1}.")

    train_items = []
    val_items = []

    for category, label in CLASS_MAPPING.items():
        cat_path = os.path.join(data_dir, category)
        if not os.path.exists(cat_path):
            print(f"[-] warning: missing directory {cat_path}")
            continue

        subdirs = _sort_subject_dirs(cat_path, category)
        if len(subdirs) < k_folds:
            raise ValueError(
                f"{category} subject count ({len(subdirs)}) is smaller than K ({k_folds})."
            )

        train_subdirs, val_subdirs, _ = _split_kfold_subjects(subdirs, k_folds, fold_idx, seed)
        train_items.extend(gather_frame_items(cat_path, train_subdirs, label))
        val_items.extend(gather_frame_items(cat_path, val_subdirs, label))

    train_rgb = {item[0] for item in train_items}
    val_rgb = {item[0] for item in val_items}
    assert train_rgb.isdisjoint(val_rgb), "train/val RGB paths overlap."
    return train_items, val_items


def load_sample(rgb_path, ir_path, augment=False):
    """이미지를 불러와 정규화한다. augment=True이면 학습용 데이터 증강을 적용한다."""
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise ValueError(f"Failed to read RGB image: {rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        raise ValueError(f"Failed to read IR image: {ir_path}")

    if augment:
        # 공간 변환: RGB/IR 동일하게 적용해 두 채널 정렬 유지
        if random.random() < 0.5:
            rgb = cv2.flip(rgb, 1)
            ir = cv2.flip(ir, 1)
        angle = random.uniform(-10, 10)
        h, w = rgb.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rgb = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR)
        ir = cv2.warpAffine(ir, M, (w, h), flags=cv2.INTER_LINEAR)

    rgb = cv2.resize(rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    ir = cv2.resize(ir, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    if augment:
        # ColorJitter (RGB only): match PyTorch after resize.
        rgb_f = rgb.astype(np.float32)
        brightness_f = random.uniform(0.7, 1.3)
        rgb_f = np.clip(rgb_f * brightness_f, 0, 255)
        contrast_f = random.uniform(0.7, 1.3)
        mean_val = rgb_f.mean()
        rgb_f = np.clip((rgb_f - mean_val) * contrast_f + mean_val, 0, 255)
        sat_f = random.uniform(0.8, 1.2)
        gray = (0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2])[:, :, np.newaxis]
        rgb_f = np.clip(gray + sat_f * (rgb_f - gray), 0, 255)
        rgb = rgb_f.astype(np.uint8)

    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - RGB_MEAN) / RGB_STD

    ir = ir.astype(np.float32) / 255.0
    ir = np.expand_dims(ir, axis=-1)
    ir = (ir - IR_MEAN) / IR_STD

    return rgb.astype(np.float32), ir.astype(np.float32)


def _generator(items, augment=False):
    for rgb_path, ir_path, label in items:
        rgb, ir = load_sample(rgb_path, ir_path, augment=augment)
        yield (rgb, ir), np.int32(label)


def make_dataset(items, batch_size=8, shuffle=False, seed=42, augment=False):
    items = list(items)
    if shuffle:
        random.Random(seed).shuffle(items)

    output_signature = (
        (
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(224, 224, 1), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda: _generator(items, augment=augment),
        output_signature=output_signature,
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(items), 2048), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def representative_dataset(items, max_samples=200):
    for rgb_path, ir_path, _ in items[:max_samples]:
        rgb, ir = load_sample(rgb_path, ir_path, augment=False)
        yield [
            np.expand_dims(rgb, axis=0).astype(np.float32),
            np.expand_dims(ir, axis=0).astype(np.float32),
        ]
