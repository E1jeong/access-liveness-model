import json

import numpy as np
import pytest
import torch
from PIL import Image

from face_detection_pipeline.data.dataset import (
    FaceDetectionDataset,
    _random_zoom_image_and_boxes,
    letterbox_image_and_boxes,
    restore_boxes_from_letterbox,
)


def _write_frame(root, *, face_rect=None, declared_size=(432, 768)):
    frame_dir = root / "live" / "high" / "subject_1" / "1"
    frame_dir.mkdir(parents=True)
    Image.new("RGB", (432, 768), color=(10, 20, 30)).save(frame_dir / "RGB.bmp")
    metadata = {
        "rgb": {
            "width": declared_size[0],
            "height": declared_size[1],
            "faceRect": [19, 566, 145, 753] if face_rect is None else face_rect,
        }
    }
    (frame_dir / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    return frame_dir


@pytest.mark.parametrize("target_size", [(320, 320), (256, 448)])
def test_letterbox_face_rect_round_trip(target_size):
    source_box = torch.tensor([[19.0, 566.0, 145.0, 753.0]])
    image = Image.new("RGB", (432, 768))

    image_tensor, transformed_box, geometry = letterbox_image_and_boxes(
        image,
        source_box,
        target_size,
    )

    assert image_tensor.shape == (3, target_size[1], target_size[0])
    assert torch.allclose(
        restore_boxes_from_letterbox(transformed_box, geometry),
        source_box,
        atol=1e-4,
    )


def test_dataset_reads_internal_faceme_box(tmp_path):
    _write_frame(tmp_path)

    dataset = FaceDetectionDataset(tmp_path, image_size=(320, 320))
    image, boxes = dataset[0]

    assert image.shape == (3, 320, 320)
    assert image.dtype == torch.float32
    assert boxes.shape == (1, 4)
    assert torch.allclose(
        boxes[0],
        torch.tensor([77.9167, 235.8333, 130.4167, 313.7500]),
        atol=1e-3,
    )


def test_dataset_accepts_explicit_empty_face_rect(tmp_path):
    _write_frame(tmp_path, face_rect=[])

    _, boxes = FaceDetectionDataset(tmp_path, image_size=(320, 320))[0]

    assert boxes.shape == (0, 4)


@pytest.mark.parametrize(
    ("face_rect", "declared_size"),
    [
        ([10, 10, 9, 20], (432, 768)),
        ([0, 0, 500, 700], (432, 768)),
        ([10, 20, 30], (432, 768)),
        ([10, 20, 30, 40], (431, 768)),
    ],
)
def test_dataset_rejects_invalid_metadata(tmp_path, face_rect, declared_size):
    _write_frame(tmp_path, face_rect=face_rect, declared_size=declared_size)
    dataset = FaceDetectionDataset(tmp_path, image_size=(320, 320))

    with pytest.raises(ValueError):
        dataset[0]


@pytest.mark.parametrize("options", [
    {"zoom_probability": -0.1}, {"zoom_probability": 1.1},
    {"zoom_probability": float("nan")},
    {"zoom_max_face_fraction": 0}, {"zoom_max_face_fraction": 1.1},
    {"zoom_max_face_fraction": float("nan")},
])
def test_dataset_rejects_invalid_zoom_options(tmp_path, options):
    _write_frame(tmp_path)
    with pytest.raises(ValueError):
        FaceDetectionDataset(tmp_path, image_size=(320, 320), **options)


def test_zoom_transforms_image_and_box_together(monkeypatch):
    monkeypatch.setattr(torch, "rand", lambda *args: torch.tensor([0.5]))
    image = Image.new("RGB", (100, 200))
    image.paste((255, 0, 0), (40, 80, 60, 120))
    original = torch.tensor([[40., 80., 60., 120.]])

    zoomed, boxes = _random_zoom_image_and_boxes(image, original, 0.9)

    assert zoomed.size == image.size
    assert torch.allclose(boxes, torch.tensor([[22.5, 45., 77.5, 155.]]), atol=1e-4)
    assert torch.equal(original, torch.tensor([[40., 80., 60., 120.]]))
    pixels = np.asarray(zoomed)
    y, x = np.where(pixels[:, :, 0] > 127)
    assert abs(x.min() - boxes[0, 0]) <= 1
    assert abs(x.max() + 1 - boxes[0, 2]) <= 1
    assert abs(y.min() - boxes[0, 1]) <= 1
    assert abs(y.max() + 1 - boxes[0, 3]) <= 1


@pytest.mark.parametrize("box", [
    [0., 0., 20., 40.], [80., 0., 100., 40.],
    [0., 160., 20., 200.], [80., 160., 100., 200.],
])
def test_zoom_keeps_edge_faces_and_avoids_padding(box):
    source = Image.new("RGB", (100, 200), color=(10, 20, 30))
    for seed in range(20):
        torch.manual_seed(seed)
        image, boxes = _random_zoom_image_and_boxes(source, torch.tensor([box]), 0.9)
        x1, y1, x2, y2 = boxes[0].tolist()
        assert -1e-4 <= x1 < x2 <= 100 + 1e-4
        assert -1e-4 <= y1 < y2 <= 200 + 1e-4
        assert max((x2-x1)/100, (y2-y1)/200) <= 0.9001
        assert max((x2-x1)/100, (y2-y1)/200) >= 0.2
        assert np.asarray(image).min() > 0


@pytest.mark.parametrize("boxes", [torch.empty((0, 4)), torch.tensor([[0., 0., 100., 200.]])])
def test_zoom_leaves_empty_and_already_large_faces_unchanged(boxes):
    source = Image.new("RGB", (100, 200))
    image, transformed = _random_zoom_image_and_boxes(source, boxes, 0.9)
    assert image is source
    assert torch.equal(transformed, boxes)


def test_dataset_zoom_is_optional_and_seeded(tmp_path):
    _write_frame(tmp_path)
    baseline = FaceDetectionDataset(tmp_path, image_size=(320, 320))
    augmented = FaceDetectionDataset(tmp_path, image_size=(320, 320), zoom_probability=1)
    before = (baseline.records[0].read_bytes(), baseline.records[0].with_name("meta.json").read_bytes())
    baseline_image, baseline_boxes = baseline[0]
    torch.manual_seed(11)
    image, boxes = augmented[0]
    torch.manual_seed(11)
    repeated_image, repeated_boxes = augmented[0]
    assert torch.equal(image, repeated_image)
    assert torch.equal(boxes, repeated_boxes)
    assert not torch.allclose(boxes, baseline_boxes)
    unchanged_image, unchanged_boxes = baseline[0]
    assert torch.equal(baseline_image, unchanged_image)
    assert torch.equal(baseline_boxes, unchanged_boxes)
    assert before == (baseline.records[0].read_bytes(), baseline.records[0].with_name("meta.json").read_bytes())


def test_zoom_preserves_all_labeled_faces():
    original = torch.tensor([[10., 20., 25., 50.], [45., 60., 60., 100.]])
    image = Image.new("RGB", (100, 200))
    for seed in range(10):
        torch.manual_seed(seed)
        _, boxes = _random_zoom_image_and_boxes(image, original, 0.9)
        assert (boxes[:, :2] >= -1e-4).all()
        assert (boxes[:, 2:] <= torch.tensor([100., 200.]) + 1e-4).all()
        scales = (boxes[:, 2:] - boxes[:, :2]) / (original[:, 2:] - original[:, :2])
        assert (scales >= 1 - 1e-4).all()
        assert torch.allclose(scales, scales[0, 0].expand_as(scales))
