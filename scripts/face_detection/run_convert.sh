#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Conversion runs on CPU without reserving the training GPU.
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=2
export PYTHONUNBUFFERED=1
.venv/bin/python -m face_detection_pipeline.export.convert "$@"
