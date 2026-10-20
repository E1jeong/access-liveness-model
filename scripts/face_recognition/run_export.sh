#!/usr/bin/env bash
# MobileNetV1 ArcFace NPU INT8 export wrapper.
set -euo pipefail
cd "$(dirname "$0")/../.."

source scripts/keras/_keras_env.sh "변환"
.venv-tf/bin/python -m face_recognition_pipeline.export.export_npu "$@"
