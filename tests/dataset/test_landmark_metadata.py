import hashlib
import json

import pytest
from PIL import Image

from common.landmark_metadata import capture_metadata_digest, load_landmark_metadata


def write_sample(root):
    frame = root / "train" / "live" / "high" / "live_1" / "1"
    frame.mkdir(parents=True)
    Image.new("RGB", (96, 128), (20, 30, 40)).save(frame / "RGB.bmp")
    metadata = {"schemaVersion": 2, "rgb": {
        "width": 96, "height": 128, "faceRect": [20, 20, 75, 110],
    }, "qualityScore": 0.0, "note": "한글", "optional": None}
    label = {
        "schemaVersion": 1, "image": "RGB.bmp", "imageWidth": 96, "imageHeight": 128,
        "coordinateSystem": "original_image_pixels", "status": "extracted",
        "imageSha256": hashlib.sha256((frame / "RGB.bmp").read_bytes()).hexdigest(),
        "captureSha256": capture_metadata_digest(metadata),
        "extractor": {"name": "FACEME", "version": "7.8.2"},
        "points": {name: {"x": x, "y": y, "useForTraining": True}
                   for name, x, y in zip(
                       ("eyeImageLeft", "eyeImageRight", "noseTip", "mouthImageLeft", "mouthImageRight"),
                       (30, 60, 45, 32, 58), (40, 40, 60, 85, 85))},
    }
    metadata["landmarks"] = label
    (frame / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    return frame, metadata, label


def test_integrated_metadata_preserves_points_and_rejects_changed_capture(tmp_path):
    frame, metadata, label = write_sample(tmp_path)
    assert load_landmark_metadata(frame / "RGB.bmp", (96, 128)) == label
    metadata["rgb"]["faceRect"][0] += 1
    (frame / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="captureSha256"):
        load_landmark_metadata(frame / "RGB.bmp", (96, 128))


def test_capture_digest_matches_android_unicode_null_and_number_fixture():
    metadata = {"schemaVersion": 2, "qualityScore": 0.0, "note": "한글", "optional": None,
                "rgb": {"width": 100, "height": 120}}
    assert capture_metadata_digest(metadata) == (
        "35592e3bd3a554c8d8bbfb897ff4f585ed917d73d42a2edb34bc2ac987b008ee"
    )


@pytest.mark.parametrize("change", ["missing", "null", "image_hash"])
def test_invalid_integrated_metadata_is_rejected(tmp_path, change):
    frame, metadata, label = write_sample(tmp_path)
    if change == "missing":
        del metadata["landmarks"]
    elif change == "null":
        metadata["landmarks"] = None
    else:
        label["imageSha256"] = "bad"
    (frame / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        load_landmark_metadata(frame / "RGB.bmp", (96, 128))


@pytest.mark.parametrize("sidecar", [True, False])
def test_reader_rejects_split_format(tmp_path, sidecar):
    frame, metadata, label = write_sample(tmp_path)
    metadata.pop("landmarks")
    metadata["schemaVersion"] = 1
    (frame / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    if sidecar:
        label["landmarks"] = label.pop("points")
        label.pop("captureSha256")
        label["metaSha256"] = hashlib.sha256((frame / "meta.json").read_bytes()).hexdigest()
        (frame / "landmarks.json").write_text(json.dumps(label), encoding="utf-8")
    with pytest.raises(ValueError, match="integrated"):
        load_landmark_metadata(frame / "RGB.bmp", (96, 128))


def test_reader_rejects_integrated_metadata_with_leftover_sidecar(tmp_path):
    frame, _, _ = write_sample(tmp_path)
    (frame / "landmarks.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="landmarks.json"):
        load_landmark_metadata(frame / "RGB.bmp", (96, 128))


def test_reader_rejects_changed_rgb_bytes(tmp_path):
    frame, _, _ = write_sample(tmp_path)
    Image.new("RGB", (96, 128), (50, 60, 70)).save(frame / "RGB.bmp")
    with pytest.raises(ValueError, match="imageSha256"):
        load_landmark_metadata(frame / "RGB.bmp", (96, 128))
