#!/usr/bin/env bash
# MobileNetV1 ArcFace Emore INT8 export with the GPU TensorFlow environment.
# Legacy wrapper: forwards to scripts/face_recognition/run_export.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

exec scripts/face_recognition/run_export.sh "$@"
