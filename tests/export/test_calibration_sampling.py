import json

import numpy as np
import pytest

from classes import CLASS_NAMES
from keras_pipeline import convert_keras_to_tflite as converter


def _items(counts):
    return [
        (f"rgb-{label}-{index}", f"ir-{label}-{index}", label)
        for label, count in enumerate(counts)
        for index in range(count)
    ]


def test_stratified_calibration_is_seeded_and_covers_every_class():
    items = _items([5, 4, 3, 2, 1, 1])

    selected_a, report_a = converter.select_stratified_calibration_items(items, 12, seed=42)
    selected_b, report_b = converter.select_stratified_calibration_items(items, 12, seed=42)

    assert selected_a == selected_b
    assert report_a == report_b
    assert all(count >= 1 for count in report_a["selected_by_class"].values())
    plentiful_counts = [
        report_a["selected_by_class"][name]
        for name in ("live", "print", "picture")
    ]
    assert max(plentiful_counts) - min(plentiful_counts) <= 1


def test_stratified_calibration_rejects_insufficient_or_missing_class_coverage():
    with pytest.raises(ValueError, match="모든 6 클래스를 포함"):
        converter.select_stratified_calibration_items(_items([1] * 6), 5, seed=42)

    with pytest.raises(ValueError, match="없는 클래스"):
        converter.select_stratified_calibration_items(_items([1, 1, 1, 1, 1, 0]), 6, seed=42)


def test_calibration_collection_uses_only_train_split(monkeypatch):
    requested_splits = []

    def fake_collect_split_items(data_dir, split):
        requested_splits.append((data_dir, split))
        return _items([1] * len(CLASS_NAMES))

    monkeypatch.setattr(converter, "collect_split_items", fake_collect_split_items)

    converter.collect_calibration_items("dataset/raw", 6, seed=42)

    assert requested_splits == [("dataset/raw", "train")]


def test_representative_generator_loads_one_item_at_a_time(monkeypatch):
    loaded = []

    def fake_load_sample(rgb_path, ir_path, augment):
        loaded.append((rgb_path, ir_path, augment))
        return np.zeros((2, 2, 3), dtype=np.float32), np.zeros((2, 2, 1), dtype=np.float32)

    monkeypatch.setattr(converter, "load_sample", fake_load_sample)
    generator = converter._make_representative_dataset_gen(_items([2, 0, 0, 0, 0, 0]), "crop_rgb")()

    assert len(loaded) == 0
    assert next(generator)[0].shape == (1, 2, 2, 3)
    assert len(loaded) == 1
    assert next(generator)[0].shape == (1, 2, 2, 3)
    assert len(loaded) == 2


def test_calibration_manifest_records_train_coverage_and_streaming_memory(tmp_path):
    items = _items([1] * len(CLASS_NAMES))
    _, report = converter.select_stratified_calibration_items(items, 6, seed=42)
    path = tmp_path / "calibration_manifest.json"

    converter.write_calibration_manifest(path, items, report, "dual", seed=42, requested_samples=6)

    manifest = json.loads(path.read_text())
    assert manifest["split"] == "train"
    assert manifest["selected_by_class"] == {name: 1 for name in CLASS_NAMES}
    assert manifest["missing_required_samples"] == 0
    assert manifest["preloaded_sample_bytes"] == 0
    assert manifest["estimated_peak_sample_bytes"] == 224 * 224 * 4 * 4
