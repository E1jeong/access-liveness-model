#!/usr/bin/env bash
# keras_pipeline/tf_model.py — 모델 구조 확인 (학습 없이 구조만 출력)
set -e
cd "$(dirname "$0")"

source scripts/_keras_env.sh "실행"

echo "=== MobileNetV2 듀얼 입력 모델 구조 출력 ==="
.venv-tf/bin/python -m keras_pipeline.tf_model "$@"
