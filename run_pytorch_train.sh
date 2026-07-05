#!/usr/bin/env bash
# pytorch_pipeline/train.py — PyTorch 모델 학습 실행기
#
# 사용 예:
#   ./run_pytorch_train.sh                          # 기본값(10 에포크, 5-Fold)
#   ./run_pytorch_train.sh --folds 5 --max-folds 1 --epochs 1
#   ./run_pytorch_train.sh --epochs 30 --batch-size 16 --learning-rate 5e-5
set -e
cd "$(dirname "$0")"

echo "=== PyTorch 학습 시작 ==="
.venv/bin/python -m pytorch_pipeline.train "$@"
