#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

.venv/bin/python -m face_detection_pipeline.training.train "$@"
