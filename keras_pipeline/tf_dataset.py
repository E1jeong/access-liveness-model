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
MULTIMODAL_INPUT_NAMES = ("cropRGB", "cropIR", "RGB", "IR", "heatmap")


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


def _normalize_rgb(rgb):
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - RGB_MEAN) / RGB_STD
    return rgb.astype(np.float32)


def _normalize_ir(ir):
    ir = ir.astype(np.float32) / 255.0
    ir = np.expand_dims(ir, axis=-1)
    ir = (ir - IR_MEAN) / IR_STD
    return ir.astype(np.float32)


def _load_rgb(path, name):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to read {name} image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_gray(path, name):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read {name} image: {path}")
    return image


def _load_heatmap(path):
    if not os.path.exists(path):
        return np.zeros(IMAGE_SIZE, dtype=np.uint8)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return np.zeros(IMAGE_SIZE, dtype=np.uint8)
    return image


def _apply_spatial_augment(image, flip, angle, interpolation):
    if flip:
        image = cv2.flip(image, 1)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=interpolation)


def _apply_rgb_jitter(image, brightness_f, contrast_f, sat_f):
    rgb_f = image.astype(np.float32)
    rgb_f = np.clip(rgb_f * brightness_f, 0, 255)
    mean_val = rgb_f.mean()
    rgb_f = np.clip((rgb_f - mean_val) * contrast_f + mean_val, 0, 255)
    gray = (0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2])[:, :, np.newaxis]
    return np.clip(gray + sat_f * (rgb_f - gray), 0, 255).astype(np.uint8)


def load_multimodal_sample(crop_rgb_path, crop_ir_path, augment=False):
    frame_dir = os.path.dirname(crop_rgb_path)
    raw_rgb_path = os.path.join(frame_dir, "RGB.bmp")
    raw_ir_path = os.path.join(frame_dir, "IR.bmp")
    heatmap_path = os.path.join(frame_dir, "face_heatmap.bmp")

    crop_rgb = _load_rgb(crop_rgb_path, "cropRGB")
    crop_ir = _load_gray(crop_ir_path, "cropIR")
    raw_rgb = _load_rgb(raw_rgb_path, "RGB")
    raw_ir = _load_gray(raw_ir_path, "IR")
    heatmap = _load_heatmap(heatmap_path)

    if augment:
        flip = random.random() < 0.5
        angle = random.uniform(-10, 10)
        crop_rgb = _apply_spatial_augment(crop_rgb, flip, angle, cv2.INTER_LINEAR)
        crop_ir = _apply_spatial_augment(crop_ir, flip, angle, cv2.INTER_LINEAR)
        raw_rgb = _apply_spatial_augment(raw_rgb, flip, angle, cv2.INTER_LINEAR)
        raw_ir = _apply_spatial_augment(raw_ir, flip, angle, cv2.INTER_LINEAR)
        heatmap = _apply_spatial_augment(heatmap, flip, angle, cv2.INTER_LINEAR)

    crop_rgb = cv2.resize(crop_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    crop_ir = cv2.resize(crop_ir, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    raw_rgb = cv2.resize(raw_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    raw_ir = cv2.resize(raw_ir, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    heatmap = cv2.resize(heatmap, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    if augment:
        brightness_f = random.uniform(0.7, 1.3)
        contrast_f = random.uniform(0.7, 1.3)
        sat_f = random.uniform(0.8, 1.2)
        crop_rgb = _apply_rgb_jitter(crop_rgb, brightness_f, contrast_f, sat_f)
        raw_rgb = _apply_rgb_jitter(raw_rgb, brightness_f, contrast_f, sat_f)

    heatmap = heatmap.astype(np.float32) / 255.0
    heatmap = np.expand_dims(heatmap, axis=-1)

    return (
        _normalize_rgb(crop_rgb),
        _normalize_ir(crop_ir),
        _normalize_rgb(raw_rgb),
        _normalize_ir(raw_ir),
        heatmap.astype(np.float32),
    )


def make_dataset(items, batch_size=8, shuffle=False, seed=42, augment=False):
    items = list(items)
    if shuffle:
        random.Random(seed).shuffle(items)

    rgb_paths = [item[0] for item in items]
    ir_paths = [item[1] for item in items]
    labels = [item[2] for item in items]

    ds = tf.data.Dataset.from_tensor_slices((rgb_paths, ir_paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(items), seed=seed, reshuffle_each_iteration=True)

    def map_fn(rgb_path, ir_path, label):
        def _py_fn(r_path, i_path, lbl):
            r_path_str = r_path.numpy().decode('utf-8')
            i_path_str = i_path.numpy().decode('utf-8')
            lbl_val = int(lbl.numpy())
            rgb, ir = load_sample(r_path_str, i_path_str, augment=augment)
            return rgb, ir, np.int32(lbl_val)

        outputs = tf.py_function(
            _py_fn,
            inp=[rgb_path, ir_path, label],
            Tout=[tf.float32, tf.float32, tf.int32]
        )
        
        outputs[0].set_shape((224, 224, 3))
        outputs[1].set_shape((224, 224, 1))
        outputs[2].set_shape(())
        
        return (outputs[0], outputs[1]), outputs[2]

    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_multimodal_dataset(items, batch_size=8, shuffle=False, seed=42, augment=False):
    items = list(items)
    if shuffle:
        random.Random(seed).shuffle(items)

    crop_rgb_paths = [item[0] for item in items]
    crop_ir_paths = [item[1] for item in items]
    labels = [item[2] for item in items]

    ds = tf.data.Dataset.from_tensor_slices((crop_rgb_paths, crop_ir_paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(items), seed=seed, reshuffle_each_iteration=True)

    def map_fn(crop_rgb_path, crop_ir_path, label):
        def _py_fn(c_rgb, c_ir, lbl):
            c_rgb_str = c_rgb.numpy().decode('utf-8')
            c_ir_str = c_ir.numpy().decode('utf-8')
            lbl_val = int(lbl.numpy())
            feats = load_multimodal_sample(c_rgb_str, c_ir_str, augment=augment)
            return feats + (np.int32(lbl_val),)

        outputs = tf.py_function(
            _py_fn,
            inp=[crop_rgb_path, crop_ir_path, label],
            Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32]
        )
        
        outputs[0].set_shape((224, 224, 3))
        outputs[1].set_shape((224, 224, 1))
        outputs[2].set_shape((224, 224, 3))
        outputs[3].set_shape((224, 224, 1))
        outputs[4].set_shape((224, 224, 1))
        outputs[5].set_shape(())
        
        return (outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]), outputs[5]

    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def representative_dataset(items, max_samples=200):
    for rgb_path, ir_path, _ in items[:max_samples]:
        rgb, ir = load_sample(rgb_path, ir_path, augment=False)
        yield [
            np.expand_dims(rgb, axis=0).astype(np.float32),
            np.expand_dims(ir, axis=0).astype(np.float32),
        ]


def representative_multimodal_dataset(items, max_samples=200):
    for crop_rgb_path, crop_ir_path, _ in items[:max_samples]:
        sample = load_multimodal_sample(crop_rgb_path, crop_ir_path, augment=False)
        yield {
            "a_crop_rgb": np.expand_dims(sample[0], axis=0).astype(np.float32),
            "b_crop_ir": np.expand_dims(sample[1], axis=0).astype(np.float32),
            "c_rgb": np.expand_dims(sample[2], axis=0).astype(np.float32),
            "d_ir": np.expand_dims(sample[3], axis=0).astype(np.float32),
            "e_heatmap": np.expand_dims(sample[4], axis=0).astype(np.float32),
        }
