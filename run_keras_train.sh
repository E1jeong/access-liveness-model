#!/usr/bin/env bash
# keras_pipeline/tf_train.py — MobileNetV2 학습
#
# 사용 예:
#   ./run_keras_train.sh                          # 기본값(fold 0, 10 에포크)
#   ./run_keras_train.sh --epochs 30 --fold-idx 1
#   ./run_keras_train.sh --folds 5 --fold-idx 0 --batch-size 16 --learning-rate 5e-5
set -e
cd "$(dirname "$0")"

source scripts/_keras_env.sh "실행됩니다"

echo "=== 학습 시작 ==="
.venv-tf/bin/python -m keras_pipeline.tf_train "$@"
