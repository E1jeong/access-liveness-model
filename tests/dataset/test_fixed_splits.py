import json
from pathlib import Path

import pytest

from common.classes import BASE_CLASS_NAMES, CLASS_NAMES
from common.utils import collect_split_items, subsample_split_items, validate_fixed_split_coverage


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
        file_path = frame_dir / filename
        file_path.write_text(str(file_path), encoding="utf-8")


def _make_conditioned_dataset(root: Path):
    for split, numbers in SPLIT_SUBJECT_NUMBERS.items():
        for category in BASE_CLASS_NAMES:
            if category == "live":
                for quality in ("high", "medium", "off"):
                    _write_frame(root / split / category / quality / f"live_{numbers[0]}")
            else:
                for environment in ("indoor", "outdoor"):
                    for number in numbers:
                        _write_frame(root / split / category / environment / f"{category}_{number}")


def test_fixed_split_validation_and_collection(tmp_path):
    _make_conditioned_dataset(tmp_path)

    counts = validate_fixed_split_coverage(tmp_path)

    expected_count = 3 + (len(BASE_CLASS_NAMES) - 1) * 4
    assert counts == {"train": expected_count, "validation": expected_count, "test": expected_count}
    assert len(collect_split_items(tmp_path, "validation")) == expected_count


def test_conditioned_layout_collects_live_off_and_other_class_environments(tmp_path):
    _make_conditioned_dataset(tmp_path)

    counts = validate_fixed_split_coverage(tmp_path)

    expected_count = 3 + (len(BASE_CLASS_NAMES) - 1) * 4
    assert counts == {"train": expected_count, "validation": expected_count, "test": expected_count}
    train_paths = [Path(rgb_path) for rgb_path, _, _ in collect_split_items(tmp_path, "train")]
    assert tmp_path / "train" / "live" / "off" / "live_1" / "1" / "cropRGB.bmp" in train_paths
    assert tmp_path / "train" / "print" / "indoor" / "print_1" / "1" / "cropRGB.bmp" in train_paths
    assert tmp_path / "train" / "print" / "outdoor" / "print_2" / "1" / "cropRGB.bmp" in train_paths


def test_conditioned_layout_rejects_unknown_condition_folder(tmp_path):
    _make_conditioned_dataset(tmp_path)
    _write_frame(tmp_path / "train" / "print" / "outdor" / "print_7")

    with pytest.raises(ValueError, match="지원하지 않는 조건 폴더"):
        collect_split_items(tmp_path, "train")


def test_rejects_mixed_legacy_and_conditioned_layout(tmp_path):
    _make_conditioned_dataset(tmp_path)
    _write_frame(tmp_path / "train" / "print" / "print_7")

    with pytest.raises(ValueError, match="평면/조건별 구조가 섞여 있습니다"):
        collect_split_items(tmp_path, "train")


def test_conditioned_spoof_folder_numbers_are_independent_across_splits(tmp_path):
    _make_conditioned_dataset(tmp_path)
    _write_frame(tmp_path / "validation" / "print" / "indoor" / "print_2")

    counts = validate_fixed_split_coverage(tmp_path)

    expected_count = 3 + (len(BASE_CLASS_NAMES) - 1) * 4
    assert counts == {
        "train": expected_count,
        "validation": expected_count + 1,
        "test": expected_count,
    }


def test_live_folder_numbers_are_independent_across_splits(tmp_path):
    _make_conditioned_dataset(tmp_path)
    _write_frame(tmp_path / "validation" / "live" / "high" / "live_1")

    counts = validate_fixed_split_coverage(tmp_path)

    expected_count = 3 + (len(BASE_CLASS_NAMES) - 1) * 4
    assert counts == {
        "train": expected_count,
        "validation": expected_count + 1,
        "test": expected_count,
    }


def test_indoor_spoof_folder_numbers_are_independent_across_splits(tmp_path):
    _make_conditioned_dataset(tmp_path)
    _write_frame(tmp_path / "validation" / "print" / "indoor" / "print_2")

    counts = validate_fixed_split_coverage(tmp_path)

    expected_count = 3 + (len(BASE_CLASS_NAMES) - 1) * 4
    assert counts == {
        "train": expected_count,
        "validation": expected_count + 1,
        "test": expected_count,
    }


def test_missing_required_file_is_rejected(tmp_path):
    _make_conditioned_dataset(tmp_path)
    missing = tmp_path / "test" / "mask" / "indoor" / "mask_5" / "1" / "IR.bmp"
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="필수 BMP 파일"):
        validate_fixed_split_coverage(tmp_path)


def test_realpath_leakage_is_rejected(tmp_path):
    _make_conditioned_dataset(tmp_path)

    src = tmp_path / "train" / "live" / "high" / "live_1" / "1" / "cropRGB.bmp"
    dst = tmp_path / "validation" / "live" / "high" / "live_3" / "1" / "cropRGB.bmp"
    dst.unlink()
    dst.symlink_to(src)

    with pytest.raises(ValueError, match="동일 프레임 실경로"):
        validate_fixed_split_coverage(tmp_path)


def test_content_hash_leakage_is_rejected(tmp_path):
    _make_conditioned_dataset(tmp_path)
    
    src = tmp_path / "train" / "live" / "high" / "live_1" / "1" / "cropRGB.bmp"
    dst = tmp_path / "validation" / "live" / "high" / "live_3" / "1" / "cropRGB.bmp"
    
    dst.write_bytes(src.read_bytes())
    
    with pytest.raises(ValueError, match="Content hash leakage detected"):
        validate_fixed_split_coverage(tmp_path)


@pytest.mark.parametrize(
    ("metadata_key", "error_message"),
    (("session_id", "Session leakage detected"), ("video_id", "Video leakage detected")),
)
def test_meta_json_session_or_video_leakage_is_rejected(
    tmp_path, metadata_key, error_message
):
    _make_conditioned_dataset(tmp_path)
    
    meta1 = tmp_path / "train" / "live" / "high" / "live_1" / "1" / "meta.json"
    meta2 = tmp_path / "validation" / "live" / "high" / "live_3" / "1" / "meta.json"
    
    with open(meta1, "w") as f:
        json.dump({metadata_key: "shared_capture_123"}, f)
    with open(meta2, "w") as f:
        json.dump({metadata_key: "shared_capture_123"}, f)
        
    with pytest.raises(ValueError, match=error_message):
        validate_fixed_split_coverage(tmp_path)


def test_meta_json_device_or_attack_medium_sharing_does_not_fail(tmp_path):
    _make_conditioned_dataset(tmp_path)

    meta1 = tmp_path / "train" / "live" / "high" / "live_1" / "1" / "meta.json"
    meta2 = tmp_path / "validation" / "live" / "high" / "live_3" / "1" / "meta.json"

    with open(meta1, "w") as f:
        json.dump({"device_id": "shared_device_A", "attack_medium": "print"}, f)
    with open(meta2, "w") as f:
        json.dump({"device_id": "shared_device_A", "attack_medium": "print"}, f)

    counts = validate_fixed_split_coverage(tmp_path)
    expected_count = 3 + (len(BASE_CLASS_NAMES) - 1) * 4
    assert counts == {"train": expected_count, "validation": expected_count, "test": expected_count}


def test_malformed_meta_json_is_rejected(tmp_path):
    _make_conditioned_dataset(tmp_path)

    meta = tmp_path / "train" / "live" / "high" / "live_1" / "1" / "meta.json"
    meta.write_text("{ not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="meta.json이 손상되었습니다"):
        validate_fixed_split_coverage(tmp_path)


def test_non_object_meta_json_is_rejected(tmp_path):
    _make_conditioned_dataset(tmp_path)

    meta = tmp_path / "train" / "live" / "high" / "live_1" / "1" / "meta.json"
    meta.write_text('["session_1"]', encoding="utf-8")

    with pytest.raises(ValueError, match="meta.json은 JSON object여야 합니다"):
        validate_fixed_split_coverage(tmp_path)


def test_subsample_split_items_is_representative_and_deterministic():
    items = [
        (f"rgb-{label}-{index}", f"ir-{label}-{index}", label)
        for label in range(len(CLASS_NAMES))
        for index in range(10)
    ]

    sampled_a = subsample_split_items(items, 24)
    sampled_b = subsample_split_items(items, 24)

    assert sampled_a == sampled_b
    assert len(sampled_a) == 24
    assert {label for _, _, label in sampled_a} == set(range(len(CLASS_NAMES)))
    assert subsample_split_items(items, len(items)) == items


def test_optional_test_split_can_be_absent(tmp_path):
    import shutil

    _make_conditioned_dataset(tmp_path)
    shutil.rmtree(tmp_path / "test")

    counts = validate_fixed_split_coverage(tmp_path)

    expected_count = 3 + (len(BASE_CLASS_NAMES) - 1) * 4
    assert counts == {"train": expected_count, "validation": expected_count, "test": 0}


def test_ensure_evaluatable_split_rejects_placeholder_test_split():
    from common.evaluate_tflite import _ensure_evaluatable_split

    placeholder_items = [(f"rgb-{i}", f"ir-{i}", i) for i in range(len(CLASS_NAMES))]
    with pytest.raises(ValueError, match="placeholder"):
        _ensure_evaluatable_split(placeholder_items, "test")

    # validation split은 소규모 fixture에서도 허용하고, test는 클래스 수 초과 시 허용한다.
    _ensure_evaluatable_split(placeholder_items, "validation")
    _ensure_evaluatable_split(placeholder_items + [("rgb-extra", "ir-extra", 0)], "test")


def test_environment_conditions_map_to_distinct_output_labels(tmp_path):
    _make_conditioned_dataset(tmp_path)
    items = collect_split_items(tmp_path, "train")
    assert {label for _, _, label in items} == set(range(24))
    for rgb_path, _, label in items:
        path = Path(rgb_path)
        category, condition = path.relative_to(tmp_path / "train").parts[:2]
        base_index = CLASS_NAMES.index(category)
        outdoor = condition in ("outdoor", "off")
        assert label == base_index + (12 if outdoor else 0)


def test_required_split_rejects_missing_outdoor_class(tmp_path):
    import shutil

    _make_conditioned_dataset(tmp_path)
    shutil.rmtree(tmp_path / "train" / "print" / "outdoor")
    with pytest.raises(ValueError, match="print_outdoor"):
        validate_fixed_split_coverage(tmp_path)


def test_legacy_flat_non_live_samples_keep_indoor_labels(tmp_path):
    import shutil

    _make_conditioned_dataset(tmp_path)
    print_root = tmp_path / "train" / "print"
    shutil.rmtree(print_root)
    _write_frame(print_root / "print_1")
    items = collect_split_items(tmp_path, "train")
    assert [label for rgb, _, label in items if Path(rgb).is_relative_to(print_root)] == [1]
