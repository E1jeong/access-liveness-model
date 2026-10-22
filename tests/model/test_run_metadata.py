import json

from antispoofing_pipeline.keras.training import run_metadata


def test_run_metadata_records_config_checkpoint_class_map_and_split_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_metadata,
        "collect_split_items",
        lambda _data_dir, split: [(f"/data/{split}/rgb.bmp", f"/data/{split}/ir.bmp", 0)],
    )
    path = tmp_path / "run.json"
    saved = run_metadata.write_run_metadata(
        path,
        "run-1",
        {"model_type": "crop_ir", "seed": 42},
        "/data",
        "model/best.keras",
        {"acer": 0.01},
    )

    assert json.loads(path.read_text()) == saved
    assert saved["run_id"] == "run-1"
    assert saved["class_map"][0] == "live"
    assert set(saved["split_hashes"]) == {"train", "validation", "test"}
    assert saved["best_checkpoint"] == "model/best.keras"


def test_split_hashes_allows_absent_or_empty_optional_test_dir(tmp_path, monkeypatch):
    (tmp_path / "train").mkdir()
    (tmp_path / "validation").mkdir()

    def fake_collect(data_dir, split):
        if split == "test":
            raise FileNotFoundError("no test split")
        return [(str(tmp_path / split / "rgb.bmp"), str(tmp_path / split / "ir.bmp"), 0)]

    monkeypatch.setattr(run_metadata, "collect_split_items", fake_collect)
    hashes_absent = run_metadata.split_hashes(tmp_path)
    assert set(hashes_absent) == {"train", "validation", "test"}

    (tmp_path / "test").mkdir()
    hashes_empty = run_metadata.split_hashes(tmp_path)
    assert hashes_empty == hashes_absent
