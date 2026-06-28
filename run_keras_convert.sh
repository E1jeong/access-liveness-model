#!/usr/bin/env bash
# keras_pipeline/convert_h5_to_tflite.py — TFLite 변환 (float / INT8)
#
# 사용 예:
#   ./run_keras_convert.sh --float --int8                    # 기본 경로 모델 변환
#   ./run_keras_convert.sh --float --int8 --fold-idx 1       # fold 1 모델 변환
#   ./run_keras_convert.sh --float                           # float 전용
#   ./run_keras_convert.sh --int8 --calibration-samples 300 # INT8 전용, 샘플 수 조정
set -e
cd "$(dirname "$0")"

export LD_LIBRARY_PATH="$(find .venv-tf/lib -path "*/nvidia/*/lib" -type d | tr '\n' ':')"

echo "=== GPU 상태 확인 ==="
.venv-tf/bin/python - <<'EOF'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU: {gpus if gpus else '없음 (CPU로 변환됩니다)'}")
EOF

echo ""
echo "=== TFLite 변환 시작 ==="
.venv-tf/bin/python -m keras_pipeline.convert_h5_to_tflite "$@"
