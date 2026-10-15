"""Face-detection model definitions."""

from face_detection_pipeline.models.yunet import YuNet, build_yunet_n

__all__ = ["YuNet", "build_yunet_n"]
