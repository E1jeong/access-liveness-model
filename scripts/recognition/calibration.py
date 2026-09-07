"""Representative face-image sampling for the recognition INT8 export."""

from collections import Counter
from pathlib import Path
import random

import cv2
import numpy as np


CALIBRATION_SAMPLES = 500
MIN_CALIBRATION_SAMPLES = 300
CALIBRATION_SPLITS = ("train", "validation")
IMAGE_NAME = "cropRGB.bmp"


def collect_live_face_paths(dataset_root):
    """Collect ``cropRGB.bmp`` paths, grouped by ``live_<subjectId>`` directory."""
    dataset_root = Path(dataset_root)
    samples_by_subject = {}

    for split in CALIBRATION_SPLITS:
        live_dir = dataset_root / split / "live"
        if not live_dir.is_dir():
            raise FileNotFoundError(f"Live calibration directory not found: {live_dir}")

        for image_path in sorted(live_dir.rglob(IMAGE_NAME)):
            subject_id = next(
                (part for part in image_path.parts if part.startswith("live_")),
                None,
            )
            if subject_id is None:
                continue
            samples_by_subject.setdefault(subject_id, []).append(str(image_path))

    if not samples_by_subject:
        raise ValueError(
            f"No {IMAGE_NAME} files under live_<subjectId> directories in "
            f"{dataset_root}"
        )
    return samples_by_subject


def select_stratified_face_paths(
    samples_by_subject,
    sample_count=CALIBRATION_SAMPLES,
    *,
    min_samples=MIN_CALIBRATION_SAMPLES,
    seed=42,
):
    """Return a deterministic, round-robin sample balanced across subjects."""
    if not min_samples <= sample_count <= CALIBRATION_SAMPLES:
        raise ValueError(
            f"Calibration sample count must be between {min_samples} and "
            f"{CALIBRATION_SAMPLES}: {sample_count}"
        )

    total_samples = sum(len(paths) for paths in samples_by_subject.values())
    if total_samples < min_samples:
        raise ValueError(
            f"At least {min_samples} live face images are required for calibration; "
            f"found {total_samples}"
        )

    rng = random.Random(seed)
    subject_paths = {
        subject_id: sorted(paths)
        for subject_id, paths in sorted(samples_by_subject.items())
        if paths
    }
    for paths in subject_paths.values():
        rng.shuffle(paths)

    selected = []
    positions = {subject_id: 0 for subject_id in subject_paths}
    while len(selected) < min(sample_count, total_samples):
        available = [
            subject_id
            for subject_id, paths in subject_paths.items()
            if positions[subject_id] < len(paths)
        ]
        if not available:
            break
        rng.shuffle(available)
        for subject_id in available:
            if len(selected) == min(sample_count, total_samples):
                break
            selected.append(subject_paths[subject_id][positions[subject_id]])
            positions[subject_id] += 1

    selected_by_subject = Counter()
    for image_path in selected:
        subject_id = next(part for part in Path(image_path).parts if part.startswith("live_"))
        selected_by_subject[subject_id] += 1
    return selected, dict(sorted(selected_by_subject.items()))


def load_normalized_face_image(image_path):
    """Load BGR BMP input as RGB and apply the recognition model preprocessing."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read calibration image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (112, 112))
    return ((image.astype(np.float32) - 127.5) / 128.0)[None, ...]
