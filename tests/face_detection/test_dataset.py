import json

import pytest
import torch
from PIL import Image

from face_detection_pipeline.data.dataset import (
    FaceDetectionDataset,
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
