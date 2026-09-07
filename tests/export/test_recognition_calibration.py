import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "recognition" / "calibration.py"
)
SPEC = importlib.util.spec_from_file_location("recognition_calibration", MODULE_PATH)
calibration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration)


def _face_tree(root, split, subject_id, frame_count):
    for frame in range(frame_count):
        image_path = root / split / "live" / "high" / subject_id / str(frame) / "cropRGB.bmp"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.touch()


def test_collects_train_and_validation_live_subject_images(tmp_path):
    _face_tree(tmp_path, "train", "live_001", 2)
    _face_tree(tmp_path, "validation", "live_002", 1)
    (tmp_path / "train" / "live" / "high" / "ignored" / "0").mkdir(parents=True)

    collected = calibration.collect_live_face_paths(tmp_path)

    assert {subject: len(paths) for subject, paths in collected.items()} == {
        "live_001": 2,
        "live_002": 1,
    }


def test_stratified_selection_is_seeded_and_balances_subjects():
    samples = {
        "live_001": [f"/live_001/{index}/cropRGB.bmp" for index in range(4)],
        "live_002": [f"/live_002/{index}/cropRGB.bmp" for index in range(4)],
        "live_003": [f"/live_003/{index}/cropRGB.bmp" for index in range(4)],
    }

    selected_a, report_a = calibration.select_stratified_face_paths(
        samples, 6, min_samples=1, seed=42
    )
    selected_b, report_b = calibration.select_stratified_face_paths(
        samples, 6, min_samples=1, seed=42
    )

    assert selected_a == selected_b
    assert report_a == report_b
    assert max(report_a.values()) - min(report_a.values()) <= 1


def test_selection_rejects_fewer_than_300_images_by_default():
    with pytest.raises(ValueError, match="At least 300"):
        calibration.select_stratified_face_paths({"live_001": ["one"] * 299})


def test_loader_resizes_bgr_bmp_to_normalized_rgb_input(tmp_path):
    image_path = tmp_path / "cropRGB.bmp"
    assert cv2.imwrite(str(image_path), np.full((224, 224, 3), (0, 0, 255), np.uint8))

    loaded = calibration.load_normalized_face_image(str(image_path))

    assert loaded.shape == (1, 112, 112, 3)
    np.testing.assert_allclose(loaded[0, 0, 0], [(255 - 127.5) / 128, -127.5 / 128, -127.5 / 128])
