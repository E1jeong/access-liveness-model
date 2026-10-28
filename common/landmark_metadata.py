"""Read version-2 capture metadata with embedded landmarks."""

from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path


LANDMARK_NAMES = (
    "eyeImageLeft", "eyeImageRight", "noseTip", "mouthImageLeft", "mouthImageRight",
)


def _canonical_json(value):
    """Sorted ASCII JSON, with finite numbers in plain, shortest decimal form."""
    if isinstance(value, dict):
        return "{" + ",".join(
            json.dumps(key, ensure_ascii=True) + ":" + _canonical_json(value[key])
            for key in sorted(value, key=lambda key: key.encode("utf-16-be", "surrogatepass"))
        ) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_canonical_json(item) for item in value) + "]"
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Capture metadata contains a non-finite number")
        number = format(Decimal(str(value)), "f")
        number = number.rstrip("0").rstrip(".") if "." in number else number
        return "0" if number == "-0" else number
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def capture_metadata_digest(metadata):
    capture = {key: value for key, value in metadata.items() if key != "landmarks"}
    return hashlib.sha256(_canonical_json(capture).encode("ascii")).hexdigest()


def load_landmark_metadata(image_path: Path, image_size: tuple[int, int]):
    """Require integrated metadata; reject separate or mixed landmark storage."""
    path = image_path.with_name("meta.json")
    sidecar = image_path.with_name("landmarks.json")
    if sidecar.exists() or sidecar.is_symlink():
        raise ValueError(f"Separate landmarks.json is unsupported; use integrated meta.json: {sidecar}")
    metadata = json.loads(path.read_bytes())
    if not isinstance(metadata, dict):
        raise ValueError(f"Capture metadata must be an object: {path}")
    label = metadata.get("landmarks")
    if (type(metadata.get("schemaVersion")) is not int or metadata["schemaVersion"] != 2
            or not isinstance(label, dict)):
        raise ValueError(f"Expected integrated meta.json version 2 with landmarks: {path}")
    if label.get("captureSha256") != capture_metadata_digest(metadata):
        raise ValueError(f"Landmark captureSha256 provenance mismatch: {path}")
    if (
        type(label.get("schemaVersion")) is not int or label["schemaVersion"] != 1
        or label.get("image") != "RGB.bmp"
        or label.get("coordinateSystem") != "original_image_pixels"
        or (label.get("imageWidth"), label.get("imageHeight")) != image_size
    ):
        raise ValueError(f"Invalid landmark coordinate contract: {path}")
    rgb = metadata.get("rgb")
    if not isinstance(rgb, dict) or (rgb.get("width"), rgb.get("height")) != image_size:
        raise ValueError(f"Capture dimensions do not match the RGB image: {path}")
    if label.get("imageSha256") != hashlib.sha256(image_path.read_bytes()).hexdigest():
        raise ValueError(f"Landmark imageSha256 provenance mismatch: {path}")
    records = label.get("points")
    if not isinstance(records, dict) or set(records) != set(LANDMARK_NAMES):
        raise ValueError(f"Expected five named landmarks: {path}")
    for name, point in records.items():
        if not isinstance(point, dict) or type(point.get("useForTraining")) is not bool:
            raise ValueError(f"Invalid landmark training mask for {name}: {path}")
        if not point["useForTraining"]:
            continue
        xy = (point.get("x"), point.get("y"))
        if label.get("status") != "extracted" or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in xy
        ):
            raise ValueError(f"Invalid selected landmark coordinate for {name}: {path}")
        if not (0 <= xy[0] < image_size[0] and 0 <= xy[1] < image_size[1]):
            raise ValueError(f"Selected landmark outside RGB image for {name}: {path}")
    return label
