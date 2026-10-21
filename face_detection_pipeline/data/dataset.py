"""Adapter for internal RGB frames and FaceMe ``rgb.faceRect`` pre-labels."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


LANDMARK_NAMES = (
    "eyeImageLeft", "eyeImageRight", "noseTip", "mouthImageLeft", "mouthImageRight",
)


@dataclass(frozen=True)
class LetterboxGeometry:
    scale: float
    pad_left: int
    pad_top: int
    source_width: int
    source_height: int
    target_width: int
    target_height: int


class FaceDetectionDataset(Dataset):
    """Load original ``RGB.bmp`` frames without changing the fixed split."""

    def __init__(
        self,
        split_dir: str | Path,
        *,
        image_size: tuple[int, int],
        limit_samples: int | None = None,
        zoom_probability: float = 0.0,
        zoom_max_face_fraction: float = 0.9,
        landmarks: bool = False,
    ):
        self.split_dir = Path(split_dir)
        self.image_size = image_size
        self.landmarks = landmarks
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Split directory does not exist: {self.split_dir}")
        _validate_target_size(image_size)
        if not 0 <= zoom_probability <= 1:
            raise ValueError("zoom_probability must be in [0, 1]")
        if not 0 < zoom_max_face_fraction <= 1:
            raise ValueError("zoom_max_face_fraction must be in (0, 1]")
        self.zoom_probability = zoom_probability
        self.zoom_max_face_fraction = zoom_max_face_fraction
        if limit_samples is not None and limit_samples <= 0:
            raise ValueError("limit_samples must be positive")

        records = sorted(self.split_dir.rglob("RGB.bmp"))
        if limit_samples is not None:
            records = records[:limit_samples]
        if not records:
            raise ValueError(f"No RGB.bmp frames found under {self.split_dir}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        image_path = self.records[index]
        meta_path = image_path.with_name("meta.json")
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing metadata for {image_path}: {meta_path}")

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        boxes = _load_face_boxes(meta_path, image.size)
        if self.landmarks:
            points, mask = _load_landmarks(image_path, image.size, boxes.shape[0])
        if self.zoom_probability > 0 and torch.rand(1).item() < self.zoom_probability:
            if self.landmarks:
                image, boxes, points = _random_zoom_image_and_boxes(
                    image, boxes, self.zoom_max_face_fraction, landmarks=points,
                )
                width, height = image.size
                mask &= ((points >= 0) & (points < points.new_tensor([width, height]))).all(-1)
            else:
                image, boxes = _random_zoom_image_and_boxes(
                    image, boxes, self.zoom_max_face_fraction,
                )
        tensor, transformed_boxes, geometry = letterbox_image_and_boxes(
            image,
            boxes,
            self.image_size,
        )
        if self.landmarks:
            points = points * geometry.scale + points.new_tensor(
                [geometry.pad_left, geometry.pad_top]
            )
            points[~mask] = 0
            return tensor, transformed_boxes, points, mask
        return tensor, transformed_boxes


def _load_landmarks(
    image_path: Path, image_size: tuple[int, int], box_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    points = torch.zeros((box_count, 5, 2), dtype=torch.float32)
    mask = torch.zeros((box_count, 5), dtype=torch.bool)
    path = image_path.with_name("landmarks.json")
    if not path.is_file():
        return points, mask
    with path.open(encoding="utf-8") as stream:
        label = json.load(stream)
    if not isinstance(label, dict) or (
        label.get("schemaVersion") != 1
        or label.get("image") != "RGB.bmp"
        or label.get("coordinateSystem") != "original_image_pixels"
        or (label.get("imageWidth"), label.get("imageHeight")) != image_size
    ):
        raise ValueError(f"Invalid landmark coordinate contract: {path}")
    for key, source in (("imageSha256", image_path),
                        ("metaSha256", image_path.with_name("meta.json"))):
        if label.get(key) != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError(f"Landmark {key} provenance mismatch: {path}")
    records = label.get("landmarks")
    if not isinstance(records, dict) or set(records) != set(LANDMARK_NAMES):
        raise ValueError(f"Expected five named landmarks: {path}")
    for index, name in enumerate(LANDMARK_NAMES):
        point = records[name]
        if not isinstance(point, dict) or type(point.get("useForTraining")) is not bool:
            raise ValueError(f"Invalid landmark training mask for {name}: {path}")
        if not point["useForTraining"]:
            continue
        if box_count != 1 or label.get("status") != "extracted":
            raise ValueError(f"Selected landmark requires an extracted face box: {path}")
        xy = (point.get("x"), point.get("y"))
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in xy):
            raise ValueError(f"Invalid selected landmark coordinate for {name}: {path}")
        if not (0 <= xy[0] < image_size[0] and 0 <= xy[1] < image_size[1]):
            raise ValueError(f"Selected landmark outside RGB image for {name}: {path}")
        points[0, index] = torch.tensor(xy)
        mask[0, index] = True
    return points, mask


def _random_zoom_image_and_boxes(
    image: Image.Image,
    boxes: torch.Tensor,
    max_face_fraction: float,
    *,
    landmarks: torch.Tensor | None = None,
):
    """Zoom a crop without padding or cutting any labeled face; preserve aspect ratio."""

    if not boxes.numel():
        return (image, boxes) if landmarks is None else (image, boxes, landmarks)
    width, height = image.size
    left, top = boxes[:, :2].amin(dim=0).tolist()
    right, bottom = boxes[:, 2:].amax(dim=0).tolist()
    fraction = max((right - left) / width, (bottom - top) / height)
    if fraction >= max_face_fraction:
        return (image, boxes) if landmarks is None else (image, boxes, landmarks)

    scale = 1 + torch.rand(1).item() * (max_face_fraction / fraction - 1)
    crop_width, crop_height = width / scale, height / scale
    min_x, max_x = max(0.0, right - crop_width), min(left, width - crop_width)
    min_y, max_y = max(0.0, bottom - crop_height), min(top, height - crop_height)
    crop_x = min_x + torch.rand(1).item() * (max_x - min_x)
    crop_y = min_y + torch.rand(1).item() * (max_y - min_y)
    zoomed = image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1 / scale, 0, crop_x, 0, 1 / scale, crop_y),
        Image.Resampling.BILINEAR,
    )
    transformed = boxes.clone()
    transformed[:, 0::2] = (transformed[:, 0::2] - crop_x) * scale
    transformed[:, 1::2] = (transformed[:, 1::2] - crop_y) * scale
    if landmarks is not None:
        transformed_points = (landmarks - landmarks.new_tensor([crop_x, crop_y])) * scale
        return zoomed, transformed, transformed_points
    return zoomed, transformed


def _load_face_boxes(meta_path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    with meta_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)

    try:
        rgb = metadata["rgb"]
        declared_size = (int(rgb["width"]), int(rgb["height"]))
        face_rect = rgb["faceRect"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid RGB metadata: {meta_path}") from exc

    if declared_size != image_size:
        raise ValueError(
            f"Metadata size {declared_size} does not match image size {image_size}: "
            f"{meta_path}"
        )
    if face_rect is None or face_rect == []:
        return torch.empty((0, 4), dtype=torch.float32)
    if not isinstance(face_rect, list) or len(face_rect) != 4:
        raise ValueError(f"rgb.faceRect must contain [x1, y1, x2, y2]: {meta_path}")

    try:
        box = torch.tensor([float(value) for value in face_rect], dtype=torch.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"rgb.faceRect contains a non-numeric value: {meta_path}") from exc
    width, height = image_size
    x1, y1, x2, y2 = box.tolist()
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"rgb.faceRect is outside the RGB frame: {meta_path}")
    return box.reshape(1, 4)


def letterbox_image_and_boxes(
    image: Image.Image,
    boxes: torch.Tensor,
    target_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, LetterboxGeometry]:
    """Resize with preserved aspect ratio and center padding."""

    _validate_target_size(target_size)
    target_width, target_height = target_size
    source_width, source_height = image.size
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = round(source_width * scale)
    resized_height = round(source_height * scale)
    pad_left = (target_width - resized_width) // 2
    pad_top = (target_height - resized_height) // 2

    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", target_size)
    canvas.paste(resized, (pad_left, pad_top))
    image_array = np.asarray(canvas, dtype=np.float32)
    image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy())

    transformed_boxes = boxes.to(dtype=torch.float32).clone().reshape(-1, 4)
    if transformed_boxes.numel():
        transformed_boxes[:, 0::2] = transformed_boxes[:, 0::2] * scale + pad_left
        transformed_boxes[:, 1::2] = transformed_boxes[:, 1::2] * scale + pad_top

    geometry = LetterboxGeometry(
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
    )
    return image_tensor, transformed_boxes, geometry


def restore_boxes_from_letterbox(
    boxes: torch.Tensor,
    geometry: LetterboxGeometry,
) -> torch.Tensor:
    restored = boxes.to(dtype=torch.float32).clone().reshape(-1, 4)
    if not restored.numel():
        return restored
    restored[:, 0::2] = (restored[:, 0::2] - geometry.pad_left) / geometry.scale
    restored[:, 1::2] = (restored[:, 1::2] - geometry.pad_top) / geometry.scale
    restored[:, 0::2].clamp_(0, geometry.source_width)
    restored[:, 1::2].clamp_(0, geometry.source_height)
    return restored


def restore_landmarks_from_letterbox(
    landmarks: torch.Tensor,
    geometry: LetterboxGeometry,
) -> torch.Tensor:
    """Restore five named points to original RGB pixels without swapping point order."""

    restored = landmarks.to(dtype=torch.float32).clone().reshape(-1, 5, 2)
    restored[..., 0] = (restored[..., 0] - geometry.pad_left) / geometry.scale
    restored[..., 1] = (restored[..., 1] - geometry.pad_top) / geometry.scale
    restored[..., 0].clamp_(0, geometry.source_width)
    restored[..., 1].clamp_(0, geometry.source_height)
    return restored


def collate_face_detection_batch(
    batch: list[tuple[torch.Tensor, ...]],
) -> tuple:
    if len(batch[0]) == 4:
        images, boxes, points, masks = zip(*batch)
        return torch.stack(images), list(boxes), list(points), list(masks)
    images, boxes = zip(*batch)
    return torch.stack(images), list(boxes)


def _validate_target_size(image_size: tuple[int, int]) -> None:
    if len(image_size) != 2 or any(size <= 0 or size % 32 for size in image_size):
        raise ValueError("Target width and height must be positive multiples of 32")
