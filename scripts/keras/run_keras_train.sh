#!/usr/bin/env bash
# keras_pipeline/tf_train.py — MobileNetV2 학습
#
# dataset/raw/{train,validation,test} 고정 분할을 사용한다.
# 사용 예:
#   ./scripts/keras/run_keras_train.sh --epochs 30 --batch-size 16 --learning-rate 5e-5
set -e
cd "$(dirname "$0")/../.."

source scripts/keras/_keras_env.sh "실행됩니다"

echo "=== 학습 시작 ==="
.venv-tf/bin/python -m keras_pipeline.tf_train "$@"
