#!/usr/bin/env bash
# keras_pipeline/convert_keras_to_tflite.py — TFLite 변환 (float / INT8)
#
# 사용 예:
#   ./scripts/keras/run_keras_convert.sh --float --int8                    # 기본 경로 모델 변환
#   ./scripts/keras/run_keras_convert.sh --float                           # float 전용
#   ./scripts/keras/run_keras_convert.sh --int8 --calibration-samples 300 # INT8 전용, 샘플 수 조정
#   ./scripts/keras/run_keras_convert.sh --npu-int8                        # NPU/NNAPI 호환 INT8 (Lambda 제거, AveragePooling2D, batch=1)
# INT8 대표 데이터셋은 dataset/raw/train만 사용한다.
set -e
cd "$(dirname "$0")/../.."

source scripts/keras/_keras_env.sh "변환됩니다"

echo "=== TFLite 변환 시작 ==="
CUDA_VISIBLE_DEVICES="" .venv-tf/bin/python -m keras_pipeline.convert_keras_to_tflite "$@"
