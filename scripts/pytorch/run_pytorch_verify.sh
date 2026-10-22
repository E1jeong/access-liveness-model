#!/usr/bin/env bash
# antispoofing_pipeline/pytorch/verify_setup.py — PyTorch 개발 환경 검증 실행기
set -e
cd "$(dirname "$0")/../.."

echo "=== PyTorch 환경 검증 시작 ==="
.venv/bin/python -m antispoofing_pipeline.pytorch.verify_setup "$@"
