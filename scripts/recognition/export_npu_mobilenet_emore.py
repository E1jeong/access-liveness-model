"""Export NPU-compliant INT8 TFLite model from MobileNetV1 face recognition checkpoint.

Deprecated: Use ``face_recognition_pipeline.export.export_npu`` instead.
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path when executed directly
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from face_recognition_pipeline.export.export_npu import main

if __name__ == "__main__":
    main()
