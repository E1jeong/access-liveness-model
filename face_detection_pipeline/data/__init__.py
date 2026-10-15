"""Face-detection dataset adapters and geometry transforms."""

from face_detection_pipeline.data.dataset import (
    FaceDetectionDataset,
    LetterboxGeometry,
    collate_face_detection_batch,
    letterbox_image_and_boxes,
    restore_boxes_from_letterbox,
)

__all__ = [
    "FaceDetectionDataset",
    "LetterboxGeometry",
    "collate_face_detection_batch",
    "letterbox_image_and_boxes",
    "restore_boxes_from_letterbox",
]
