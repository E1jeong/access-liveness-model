"""Representative face-image sampling for the recognition INT8 export.

Deprecated: Use ``face_recognition_pipeline.data.calibration`` instead.
"""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from face_recognition_pipeline.data.calibration import (
    CALIBRATION_SAMPLES,
    CALIBRATION_SPLITS,
    IMAGE_NAME,
    MIN_CALIBRATION_SAMPLES,
    collect_live_face_paths,
    load_normalized_face_image,
    select_stratified_face_paths,
)

__all__ = [
    "CALIBRATION_SAMPLES",
    "CALIBRATION_SPLITS",
    "IMAGE_NAME",
    "MIN_CALIBRATION_SAMPLES",
    "collect_live_face_paths",
    "load_normalized_face_image",
    "select_stratified_face_paths",
]
