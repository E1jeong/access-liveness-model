import json

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from face_detection_pipeline.data.dataset import collate_face_detection_batch
from face_detection_pipeline.models.yunet import build_yunet_n
from face_detection_pipeline.training.train import (
    _resolve_device,
    _run_epoch,
    parse_args,
    train,
)


def _write_split_frame(data_dir, split):
    frame_dir = data_dir / split / "live" / "high" / "subject_1" / "1"
    frame_dir.mkdir(parents=True)
    Image.new("RGB", (96, 128), color=(10, 20, 30)).save(frame_dir / "RGB.bmp")
    metadata = {
        "rgb": {
            "width": 96,
            "height": 128,
            "faceRect": [20, 20, 75, 110],
        }
    }
    (frame_dir / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_one_epoch_cpu_smoke_saves_box_only_checkpoint(tmp_path, monkeypatch):
    import face_detection_pipeline.training.train as training

    datasets = []
    original_dataset = training.FaceDetectionDataset

    def capture_dataset(*args, **kwargs):
        dataset = original_dataset(*args, **kwargs)
        datasets.append(dataset)
        return dataset

    monkeypatch.setattr(training, "FaceDetectionDataset", capture_dataset)
    data_dir = tmp_path / "dataset" / "raw"
    _write_split_frame(data_dir, "train")
    _write_split_frame(data_dir, "validation")
    output_dir = tmp_path / "model"
    args = parse_args(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--run-id",
            "smoke",
            "--image-width",
            "64",
            "--image-height",
            "64",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ]
    )

    best_path = train(args)

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    assert checkpoint["model"] == "yunet_n_box_only"
    assert checkpoint["run_id"] == "smoke"
    assert checkpoint["input_size"] == [64, 64]
    assert checkpoint["training_config"]["zoom_probability"] == 0.5
    assert checkpoint["training_config"]["zoom_max_face_fraction"] == 0.9
    assert datasets[0].zoom_probability == 0.5
    assert datasets[1].zoom_probability == 0
    assert checkpoint["epoch"] == 1
    assert checkpoint["warmup_state"]["completed_iterations"] == 1
    assert checkpoint["validation_metrics"]["f1"] >= 0
    assert len(checkpoint["model_state_dict"]) > 0
    assert (output_dir / "smoke" / "latest_yunet_n_64x64.pth").is_file()
    assert (output_dir / "smoke" / "metrics_yunet_n_64x64.json").is_file()

    resumed_args = parse_args(
        [
            "--data-dir",
            str(data_dir),
            "--epochs",
            "2",
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--image-width",
            "64",
            "--image-height",
            "64",
            "--resume",
            str(output_dir / "smoke" / "latest_yunet_n_64x64.pth"),
        ]
    )
    resumed_args.zoom_probability = 0
    with pytest.raises(ValueError, match="Resume zoom_probability"):
        train(resumed_args)
    assert torch.load(
        output_dir / "smoke" / "latest_yunet_n_64x64.pth",
        map_location="cpu", weights_only=True,
    )["epoch"] == 1
    resumed_args.zoom_probability = 0.5
    train(resumed_args)
    resumed = torch.load(
        output_dir / "smoke" / "latest_yunet_n_64x64.pth",
        map_location="cpu",
        weights_only=True,
    )
    history = json.loads(
        (output_dir / "smoke" / "metrics_yunet_n_64x64.json").read_text()
    )
    assert resumed["epoch"] == 2
    assert resumed["warmup_state"]["completed_iterations"] == 2
    assert len(history) == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["--image-width", "321"],
        ["--image-height", "0"],
        ["--epochs", "0"],
        ["--batch-size", "0"],
        ["--learning-rate", "0"],
        ["--num-workers", "-1"],
        ["--warmup-iters", "-1"],
        ["--warmup-ratio", "0"],
        ["--warmup-ratio", "2"],
        ["--score-threshold", "-0.1"],
        ["--nms-threshold", "1.1"],
        ["--iou-threshold", "0"],
        ["--max-detections", "0"],
        ["--run-id", "../outside"],
        ["--zoom-probability", "-0.1"],
        ["--zoom-probability", "nan"],
        ["--zoom-max-face-fraction", "0"],
        ["--zoom-max-face-fraction", "inf"],
    ],
)
def test_train_rejects_invalid_arguments(arguments):
    with pytest.raises(ValueError):
        parse_args(arguments)


def test_training_aborts_on_non_finite_loss():
    model = build_yunet_n()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = DataLoader(
        [(torch.zeros(3, 64, 64), torch.empty((0, 4)))],
        batch_size=1,
        collate_fn=collate_face_detection_batch,
    )

    class NonFiniteCriterion:
        def __call__(self, outputs, boxes):
            infinite = outputs.cls_scores[0].sum() * 0 + float("inf")
            return {
                "total": infinite,
                "classification": infinite,
                "objectness": infinite,
                "bbox": infinite,
            }

    with pytest.raises(FloatingPointError, match="Non-finite loss"):
        _run_epoch(
            model,
            loader,
            NonFiniteCriterion(),
            optimizer,
            torch.device("cpu"),
        )


def test_resolve_device_supports_mps_when_available(monkeypatch):
    args = parse_args(["--device", "mps"])
    assert args.device == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS was requested but is not available"):
        _resolve_device("mps")

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _resolve_device("mps") == torch.device("mps")
