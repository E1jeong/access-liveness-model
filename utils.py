import os
import random
import numpy as np
from classes import CLASS_NAMES, CLASS_MAPPING


def _sort_subject_dirs(cat_path, category):
    prefix = f"{category}_"
    subdirs = [
        d for d in os.listdir(cat_path)
        if os.path.isdir(os.path.join(cat_path, d)) and d.startswith(prefix)
    ]
    try:
        return sorted(subdirs, key=lambda x: int(x[len(prefix):]))
    except ValueError:
        return sorted(subdirs)


def _sort_frame_dirs(subject_path):
    subdirs = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]
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
                f"{category} 클래스의 subject 폴더 수({len(subdirs)})가 K({k_folds})보다 적습니다."
            )

        _, _, folds = _split_kfold_subjects(subdirs, k_folds, 0, seed)
        seen = [sd for fold in folds for sd in fold]
        assert len(seen) == len(set(seen)), \
            f"{category} 클래스의 fold validation subject가 서로 겹칩니다."
        assert set(seen) == set(subdirs), \
            f"{category} 클래스의 fold validation subject가 전체 subject를 덮지 못합니다."


def gather_frame_items(cat_path, subdirs_list, label):
    """subject 폴더 목록에서 (rgb_path, ir_path, label) 튜플 리스트를 수집한다."""
    gathered = []
    for sd in subdirs_list:
        subject_path = os.path.join(cat_path, sd)
        for frame_id in _sort_frame_dirs(subject_path):
            frame_path = os.path.join(subject_path, frame_id)
            rgb_path = os.path.join(frame_path, "cropRGB.bmp")
            ir_path = os.path.join(frame_path, "cropIR.bmp")
            raw_rgb_path = os.path.join(frame_path, "RGB.bmp")
            raw_ir_path = os.path.join(frame_path, "IR.bmp")
            required = [rgb_path, ir_path, raw_rgb_path, raw_ir_path]
            if not all(os.path.exists(p) for p in required):
                raise FileNotFoundError(f"필수 BMP 파일이 누락되었습니다: {frame_path}")
            gathered.append((rgb_path, ir_path, int(label)))
    return gathered


def calculate_validation_metrics(labels, preds):
    """혼동 행렬, 클래스별 Recall, APCER/BPCER/ACER를 계산한다."""
    num_classes = len(CLASS_NAMES)
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, pred in zip(labels, preds):
        confusion_matrix[int(label), int(pred)] += 1

    recalls = []
    for class_idx in range(num_classes):
        total = confusion_matrix[class_idx, :].sum()
        correct = confusion_matrix[class_idx, class_idx]
        recalls.append(float(correct / total) if total > 0 else 0.0)

    live_mask = labels == 0
    spoof_mask = labels != 0
    total_live = int(live_mask.sum())
    total_spoof = int(spoof_mask.sum())
    apcer_errors = int(((preds == 0) & spoof_mask).sum())
    bpcer_errors = int(((preds != 0) & live_mask).sum())

    apcer = apcer_errors / total_spoof if total_spoof > 0 else 0.0
    bpcer = bpcer_errors / total_live if total_live > 0 else 0.0
    acer = (apcer + bpcer) / 2.0
    return confusion_matrix, recalls, apcer, bpcer, acer
