#!/usr/bin/env bash
# pytorch_pipeline/model.py — PyTorch 모델 구조 및 더미 출력 검증 실행기
set -e
cd "$(dirname "$0")/../.."

echo "=== PyTorch MobileNetV3 듀얼 입력 모델 구조 출력 ==="
.venv/bin/python -m pytorch_pipeline.model "$@"
