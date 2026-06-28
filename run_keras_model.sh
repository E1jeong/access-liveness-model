#!/usr/bin/env bash
# keras_pipeline/tf_model.py — 모델 구조 확인 (학습 없이 구조만 출력)
set -e
cd "$(dirname "$0")"

export LD_LIBRARY_PATH="$(find .venv-tf/lib -path "*/nvidia/*/lib" -type d | tr '\n' ':')"

echo "=== GPU 상태 확인 ==="
.venv-tf/bin/python - <<'EOF'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU: {gpus if gpus else '없음 (CPU로 실행)'}")
EOF

echo ""
echo "=== MobileNetV2 듀얼 입력 모델 구조 출력 ==="
.venv-tf/bin/python -m keras_pipeline.tf_model "$@"
