from pathlib import Path

import pytest

from classes import CLASS_NAMES
from utils import collect_split_items, validate_fixed_split_coverage


SPLIT_SUBJECT_NUMBERS = {
    "train": (1, 2),
    "validation": (3, 4),
    "test": (5, 6),
}
REQUIRED_FILES = ("cropRGB.bmp", "cropIR.bmp", "RGB.bmp", "IR.bmp")


def _write_frame(subject_dir: Path):
    frame_dir = subject_dir / "1"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_FILES:
        (frame_dir / filename).touch()


def _make_dataset(root: Path):
    for split, numbers in SPLIT_SUBJECT_NUMBERS.items():
        for category in CLASS_NAMES:
            if category == "live":
                for quality in ("high", "medium"):
                    _write_frame(root / split / category / quality / f"live_{numbers[0]}")
            else:
                for number in numbers:
                    _write_frame(root / split / category / f"{category}_{number}")


def test_fixed_split_validation_and_collection(tmp_path):
    _make_dataset(tmp_path)

    counts = validate_fixed_split_coverage(tmp_path)

    assert counts == {"train": 12, "validation": 12, "test": 12}
    assert len(collect_split_items(tmp_path, "validation")) == 12


def test_live_subject_cannot_cross_splits(tmp_path):
    _make_dataset(tmp_path)
    _write_frame(tmp_path / "validation" / "live" / "high" / "live_1")

    with pytest.raises(ValueError, match="live.*여러 split"):
        validate_fixed_split_coverage(tmp_path)


def test_spoof_physical_subject_pair_cannot_cross_splits(tmp_path):
    _make_dataset(tmp_path)
    _write_frame(tmp_path / "validation" / "print" / "print_2")

    with pytest.raises(ValueError, match="print.*여러 split"):
        validate_fixed_split_coverage(tmp_path)


def test_missing_required_file_is_rejected(tmp_path):
    _make_dataset(tmp_path)
    missing = tmp_path / "test" / "mask" / "mask_5" / "1" / "IR.bmp"
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="필수 BMP 파일"):
        validate_fixed_split_coverage(tmp_path)
