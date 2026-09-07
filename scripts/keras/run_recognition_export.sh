#!/usr/bin/env bash
# MobileNetV1 ArcFace Emore INT8 export with the GPU TensorFlow environment.
set -euo pipefail
cd "$(dirname "$0")/../.."

source scripts/keras/_keras_env.sh "변환"
.venv-tf/bin/python scripts/recognition/export_npu_mobilenet_emore.py "$@"
