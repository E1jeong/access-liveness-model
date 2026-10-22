#!/usr/bin/env bash
# antispoofing_pipeline/pytorch/train.py — PyTorch 모델 학습 실행기
#
# 사용 예:
#   ./scripts/pytorch/run_pytorch_train.sh
#   ./scripts/pytorch/run_pytorch_train.sh --epochs 30 --batch-size 16 --learning-rate 5e-5
set -e
cd "$(dirname "$0")/../.."

echo "=== PyTorch 학습 시작 ==="
.venv/bin/python -m antispoofing_pipeline.pytorch.train "$@"
