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

IMAGE_SIZE = (224, 224)
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IR_MEAN = np.array([0.5], dtype=np.float32)
IR_STD = np.array([0.5], dtype=np.float32)


def _sort_subject_dirs(cat_path, category):
    prefix = f"{category}_"
    subdirs = [
        d
        for d in os.listdir(cat_path)
        if os.path.isdir(os.path.join(cat_path, d)) and d.startswith(prefix)
    ]
    try:
        return sorted(subdirs, key=lambda x: int(x[len(prefix):]))
    except ValueError:
        return sorted(subdirs)


def _sort_frame_dirs(subject_path):
    subdirs = [
        d
        for d in os.listdir(subject_path)
        if os.path.isdir(os.path.join(subject_path, d))
    ]
    try:
        return sorted(subdirs, key=lambda x: int(x))
    except ValueError:
        return sorted(subdirs)


def _split_kfold_subjects(subdirs, k_folds, fold_idx, seed):
    shuffled = list(subdirs)
    random.Random(seed).shuffle(shuffled)
    folds = [shuffled[i::k_folds] for i in range(k_folds)]
    val_subdirs = folds[fold_idx]
    train_subdirs = [sd for i, fold in enumerate(folds) if i != fold_idx for sd in fold]
    return train_subdirs, val_subdirs, folds


def validate_kfold_coverage(data_dir="dataset/raw", k_folds=5, seed=42):
    for category in CLASS_MAPPING.keys():
        cat_path = os.path.join(data_dir, category)
        if not os.path.exists(cat_path):
            continue

        subdirs = _sort_subject_dirs(cat_path, category)
        if len(subdirs) < k_folds:
            raise ValueError(
                f"{category} subject count ({len(subdirs)}) is smaller than K ({k_folds})."
            )

        _, _, folds = _split_kfold_subjects(subdirs, k_folds, 0, seed)
        seen = [sd for fold in folds for sd in fold]
        assert len(seen) == len(set(seen)), f"{category} validation subjects overlap."
        assert set(seen) == set(subdirs), f"{category} validation subjects do not cover all subjects."


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

        def gather(subdir_list):
            gathered = []
            for sd in subdir_list:
                subject_path = os.path.join(cat_path, sd)
                for frame_id in _sort_frame_dirs(subject_path):
                    frame_path = os.path.join(subject_path, frame_id)
                    rgb_path = os.path.join(frame_path, "cropRGB.bmp")
                    ir_path = os.path.join(frame_path, "cropIR.bmp")
                    raw_rgb_path = os.path.join(frame_path, "RGB.bmp")
                    raw_ir_path = os.path.join(frame_path, "IR.bmp")
                    required = [rgb_path, ir_path, raw_rgb_path, raw_ir_path]
                    if not all(os.path.exists(path) for path in required):
                        raise FileNotFoundError(f"Missing required BMP file under {frame_path}")
                    gathered.append((rgb_path, ir_path, int(label)))
            return gathered

        train_items.extend(gather(train_subdirs))
        val_items.extend(gather(val_subdirs))

    train_rgb = {item[0] for item in train_items}
    val_rgb = {item[0] for item in val_items}
    assert train_rgb.isdisjoint(val_rgb), "train/val RGB paths overlap."
    return train_items, val_items


def load_sample(rgb_path, ir_path):
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise ValueError(f"Failed to read RGB image: {rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    rgb = (rgb - RGB_MEAN) / RGB_STD

    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        raise ValueError(f"Failed to read IR image: {ir_path}")
    ir = cv2.resize(ir, IMAGE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    ir = np.expand_dims(ir, axis=-1)
    ir = (ir - IR_MEAN) / IR_STD
    return rgb.astype(np.float32), ir.astype(np.float32)


def _generator(items):
    for rgb_path, ir_path, label in items:
        rgb, ir = load_sample(rgb_path, ir_path)
        yield (rgb, ir), np.int32(label)


def make_dataset(items, batch_size=8, shuffle=False, seed=42):
    output_signature = (
        (
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(224, 224, 1), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda: _generator(items),
        output_signature=output_signature,
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(items), 2048), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def representative_dataset(items, max_samples=200):
    for rgb_path, ir_path, _ in items[:max_samples]:
        rgb, ir = load_sample(rgb_path, ir_path)
        yield [
            np.expand_dims(rgb, axis=0).astype(np.float32),
            np.expand_dims(ir, axis=0).astype(np.float32),
        ]
