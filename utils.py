import os
import random
import numpy as np
from classes import CLASS_NAMES, CLASS_MAPPING

FIXED_SPLITS = ("train", "validation", "test")


def _subject_sort_key(subject):
    basename = os.path.basename(subject)
    try:
        return (0, int(basename.rsplit("_", 1)[1]))
    except (IndexError, ValueError):
        return (1, basename)


def _sort_subject_dirs(cat_path, category):
    prefix = f"{category}_"
    if category == "live":
        subdirs = []
        for subdir in ["high", "medium"]:
            sub_path = os.path.join(cat_path, subdir)
            if os.path.exists(sub_path):
                matched = [
                    os.path.join(subdir, d) for d in os.listdir(sub_path)
                    if os.path.isdir(os.path.join(sub_path, d)) and d.startswith(prefix)
                ]
                try:
                    matched = sorted(matched, key=lambda x: int(os.path.basename(x)[len(prefix):]))
                except ValueError:
                    matched = sorted(matched)
                subdirs.extend(matched)
        return subdirs
    else:
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


def _group_subject_dirs(subdirs, category):
    # subdirs는 이미 정렬되어 있는 상태입니다. (_sort_subject_dirs의 리턴값)
    # subdirs를 동일 인물(Group)로 묶습니다.
    groups = {}
    if category == "live":
        # live의 경우: high/live_1, medium/live_1 등이 있으므로 basename에서 숫자를 추출하여 그룹화
        for sd in subdirs:
            basename = os.path.basename(sd)  # "live_1"
            try:
                group_key = int(basename.split("_")[1])
            except (IndexError, ValueError):
                group_key = basename  # fallback
            groups.setdefault(group_key, []).append(sd)
    else:
        # 그 외 spoof: subject 목록을 정렬한 순서대로 2개씩 묶음
        for i, sd in enumerate(subdirs):
            group_key = i // 2
            groups.setdefault(group_key, []).append(sd)
    return groups


def _split_kfold_subjects(subdirs, k_folds, fold_idx, seed, category):
    groups = _group_subject_dirs(subdirs, category)

    # 그룹 키 정렬 리스트 생성 (일관성 보장)
    group_keys = sorted(list(groups.keys()))
    # 그룹 키들을 셔플합니다.
    random.Random(seed).shuffle(group_keys)

    # 그룹 키들을 k_folds로 나눕니다.
    folds_keys = [group_keys[i::k_folds] for i in range(k_folds)]

    # 각 fold의 실제 subdir 목록을 만듭니다.
    folds = []
    for f_keys in folds_keys:
        fold_subdirs = []
        for k in f_keys:
            fold_subdirs.extend(groups[k])
        folds.append(fold_subdirs)
        
    val_subdirs = folds[fold_idx]
    train_subdirs = [sd for i, fold in enumerate(folds) if i != fold_idx for sd in fold]
    return train_subdirs, val_subdirs, folds


def collect_split_items(data_dir="dataset/raw", split="train"):
    """고정 split 하나에서 (cropRGB, cropIR, label) 항목을 수집한다."""
    if split not in FIXED_SPLITS:
        raise ValueError(f"split은 {FIXED_SPLITS} 중 하나여야 합니다: {split}")

    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"고정 split 디렉터리가 없습니다: {split_dir}")

    items = []
    for category, label in CLASS_MAPPING.items():
        cat_path = os.path.join(split_dir, category)
        if not os.path.isdir(cat_path):
            raise FileNotFoundError(f"클래스 디렉터리가 없습니다: {cat_path}")

        subdirs = _sort_subject_dirs(cat_path, category)
        if not subdirs:
            raise ValueError(f"subject 폴더가 비어 있습니다: {cat_path}")

        category_items = gather_frame_items(cat_path, subdirs, label)
        if not category_items:
            raise ValueError(f"프레임이 비어 있습니다: {cat_path}")
        items.extend(category_items)

    return items


def validate_fixed_split_coverage(data_dir="dataset/raw"):
    """고정 split의 완전성과 subject/frame 누수를 검사한다.

    live는 high/medium의 동일 번호를 같은 인물로 본다. spoof 클래스는 기존
    Group K-Fold 계약과 동일하게 전체 subject 정렬 순서에서 연속 두 폴더를
    같은 물리 인물로 본다.
    """
    split_subjects = {split: {} for split in FIXED_SPLITS}
    split_items = {}

    for split in FIXED_SPLITS:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"고정 split 디렉터리가 없습니다: {split_dir}")

        for category in CLASS_MAPPING:
            cat_path = os.path.join(split_dir, category)
            if not os.path.isdir(cat_path):
                raise FileNotFoundError(f"클래스 디렉터리가 없습니다: {cat_path}")
            subdirs = _sort_subject_dirs(cat_path, category)
            if not subdirs:
                raise ValueError(f"subject 폴더가 비어 있습니다: {cat_path}")
            split_subjects[split][category] = subdirs

        split_items[split] = collect_split_items(data_dir, split)

    for category in CLASS_MAPPING:
        all_subdirs = sorted({
            subject
            for split in FIXED_SPLITS
            for subject in split_subjects[split][category]
        }, key=_subject_sort_key)
        groups = _group_subject_dirs(all_subdirs, category)
        subject_to_group = {
            subject: group_key
            for group_key, subjects in groups.items()
            for subject in subjects
        }

        group_splits = {}
        for split in FIXED_SPLITS:
            for subject in split_subjects[split][category]:
                group_key = subject_to_group[subject]
                group_splits.setdefault(group_key, set()).add(split)

        overlaps = {
            group_key: sorted(splits)
            for group_key, splits in group_splits.items()
            if len(splits) > 1
        }
        if overlaps:
            raise ValueError(
                f"{category} 클래스의 동일 물리 subject가 여러 split에 있습니다: {overlaps}"
            )

    resolved_paths = {}
    for split, items in split_items.items():
        paths = {
            os.path.realpath(path)
            for rgb_path, ir_path, _ in items
            for path in (rgb_path, ir_path)
        }
        for other_split, other_paths in resolved_paths.items():
            overlap = paths & other_paths
            if overlap:
                example = sorted(overlap)[0]
                raise ValueError(
                    f"{other_split}/{split} split에 동일 프레임 실경로가 있습니다: {example}"
                )
        resolved_paths[split] = paths

    return {split: len(items) for split, items in split_items.items()}


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

        _, _, folds = _split_kfold_subjects(subdirs, k_folds, 0, seed, category)
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


def quantize_for_tflite(arr, detail):
    """int8/uint8-quantize `arr` per an interpreter tensor `detail` dict
    (from `interpreter.get_input_details()`); float dtype is a no-op cast."""
    dtype = detail['dtype']
    if dtype not in (np.int8, np.uint8):
        return arr.astype(np.float32)
    scale, zero_point = detail['quantization']
    if scale == 0.0:
        scale = 1.0
    q = np.round(arr / scale) + zero_point
    info = np.iinfo(dtype)
    return np.clip(q, info.min, info.max).astype(dtype)


def dequantize_from_tflite(arr, detail):
    dtype = detail['dtype']
    if dtype not in (np.int8, np.uint8):
        return arr.astype(np.float32)
    scale, zero_point = detail['quantization']
    return (arr.astype(np.float32) - zero_point) * scale


def calculate_validation_metrics(labels, preds):
    """혼동 행렬, 클래스별 Recall, APCER/BPCER/ACER를 계산한다."""
    num_classes = len(CLASS_NAMES)
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)

    if labels.ndim != 1 or preds.ndim != 1:
        raise ValueError("labels와 preds는 1차원이어야 합니다.")
    if len(labels) != len(preds):
        raise ValueError("labels와 preds의 길이가 같아야 합니다.")
    if np.any((labels < 0) | (labels >= num_classes)):
        raise ValueError(f"labels는 0 이상 {num_classes - 1} 이하여야 합니다.")
    if np.any((preds < 0) | (preds >= num_classes)):
        raise ValueError(f"preds는 0 이상 {num_classes - 1} 이하여야 합니다.")

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
