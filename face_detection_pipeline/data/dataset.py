"""Adapter for internal RGB frames and FaceMe ``rgb.faceRect`` pre-labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


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
    ):
        self.split_dir = Path(split_dir)
        self.image_size = image_size
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Split directory does not exist: {self.split_dir}")
        _validate_target_size(image_size)
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.records[index]
        meta_path = image_path.with_name("meta.json")
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing metadata for {image_path}: {meta_path}")

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        boxes = _load_face_boxes(meta_path, image.size)
        tensor, transformed_boxes, _ = letterbox_image_and_boxes(
            image,
            boxes,
            self.image_size,
        )
        return tensor, transformed_boxes


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


def collate_face_detection_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    images, boxes = zip(*batch)
    return torch.stack(images), list(boxes)


def _validate_target_size(image_size: tuple[int, int]) -> None:
    if len(image_size) != 2 or any(size <= 0 or size % 32 for size in image_size):
        raise ValueError("Target width and height must be positive multiples of 32")
