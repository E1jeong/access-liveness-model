from pathlib import Path
import cv2
import numpy as np
import pytest

from face_recognition_pipeline.data.calibration import (
    CALIBRATION_SAMPLES,
    collect_live_face_paths,
    load_normalized_face_image,
    select_stratified_face_paths,
)


def _face_tree(root, split, category, subject_id, frame_count):
    for frame in range(frame_count):
        image_path = root / split / category / "high" / subject_id / str(frame) / "cropRGB.bmp"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.touch()


def test_collects_train_and_validation_live_subject_images(tmp_path):
    _face_tree(tmp_path, "train", "live", "live_001", 2)
    _face_tree(tmp_path, "validation", "live", "live_002", 1)
    (tmp_path / "train" / "live" / "high" / "ignored" / "0").mkdir(parents=True)

    collected = collect_live_face_paths(tmp_path)

    assert {subject: len(paths) for subject, paths in collected.items()} == {
        "live_001": 2,
        "live_002": 1,
    }


def test_collect_ignores_spoof_folders_and_requires_live_split(tmp_path):
    # Setup live in train and validation
    _face_tree(tmp_path, "train", "live", "live_001", 1)
    _face_tree(tmp_path, "validation", "live", "live_002", 1)
    # Setup spoof folders in train
    _face_tree(tmp_path, "train", "spoof_screen", "live_003", 5)
    _face_tree(tmp_path, "train", "spoof_paper", "live_004", 5)

    collected = collect_live_face_paths(tmp_path)

    assert "live_003" not in collected
    assert "live_004" not in collected
    assert set(collected.keys()) == {"live_001", "live_002"}


def test_collect_raises_when_live_split_missing(tmp_path):
    (tmp_path / "train" / "live").mkdir(parents=True)
    # validation/live missing
    with pytest.raises(FileNotFoundError, match="Live calibration directory not found"):
        collect_live_face_paths(tmp_path)


def test_stratified_selection_is_seeded_and_balances_subjects():
    samples = {
        "live_001": [f"/live_001/{index}/cropRGB.bmp" for index in range(4)],
        "live_002": [f"/live_002/{index}/cropRGB.bmp" for index in range(4)],
        "live_003": [f"/live_003/{index}/cropRGB.bmp" for index in range(4)],
    }

    selected_a, report_a = select_stratified_face_paths(
        samples, 6, min_samples=1, seed=42
    )
    selected_b, report_b = select_stratified_face_paths(
        samples, 6, min_samples=1, seed=42
    )

    assert selected_a == selected_b
    assert report_a == report_b
    assert max(report_a.values()) - min(report_a.values()) <= 1


def test_selection_rejects_out_of_range_sample_count():
    with pytest.raises(ValueError, match="At least 300"):
        select_stratified_face_paths({"live_001": ["one"] * 299})

    with pytest.raises(ValueError, match="must be between"):
        select_stratified_face_paths({"live_001": ["one"] * 600}, sample_count=501)


def test_load_normalized_face_image_converts_bgr_to_rgb(tmp_path):
    # Pure red in BGR: B=0, G=0, R=255
    red_bgr = np.zeros((120, 120, 3), dtype=np.uint8)
    red_bgr[:, :, 2] = 255
    image_path = tmp_path / "cropRGB.bmp"
    cv2.imwrite(str(image_path), red_bgr)

    normalized = load_normalized_face_image(image_path)

    assert normalized.shape == (1, 112, 112, 3)
    assert normalized.dtype == np.float32

    # In RGB: channel 0 is R (255 -> ~0.996), channels 1 & 2 are G, B (0 -> ~ -0.996)
    expected_red = (255.0 - 127.5) / 128.0
    expected_zero = (0.0 - 127.5) / 128.0
    assert np.allclose(normalized[0, :, :, 0], expected_red, atol=1e-3)
    assert np.allclose(normalized[0, :, :, 1], expected_zero, atol=1e-3)
    assert np.allclose(normalized[0, :, :, 2], expected_zero, atol=1e-3)
