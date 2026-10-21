#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Prevent macOS low default soft file-descriptor limit (256) from exhausting DataLoader pipes.
ulimit -n 65536 2>/dev/null || ulimit -n 10240 2>/dev/null || true

.venv/bin/python -m face_detection_pipeline.training.train "$@"
