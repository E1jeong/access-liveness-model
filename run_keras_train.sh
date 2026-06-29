#!/usr/bin/env bash
# keras_pipeline/train_tf.py — MobileNetV2 학습
#
# 사용 예:
#   ./run_keras_train.sh                          # 기본값(fold 0, 10 에포크)
#   ./run_keras_train.sh --epochs 30 --fold-idx 1
#   ./run_keras_train.sh --folds 5 --fold-idx 0 --batch-size 16 --learning-rate 5e-5
set -e
cd "$(dirname "$0")"

export LD_LIBRARY_PATH="$(find .venv-tf/lib -path "*/nvidia/*/lib" -type d | tr '\n' ':')"

echo "=== GPU 상태 확인 ==="
.venv-tf/bin/python - <<'EOF'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU: {gpus if gpus else '없음 (CPU로 실행됩니다)'}")
EOF

echo ""
echo "=== 학습 시작 ==="
.venv-tf/bin/python -m keras_pipeline.train_tf "$@"
