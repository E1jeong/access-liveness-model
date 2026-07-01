#!/usr/bin/env bash
# Train the Keras 5-input multimodal MobileNetV2 pipeline.
#
# Examples:
#   ./run_keras_train.sh
#   ./run_keras_train.sh --epochs 30 --fold-idx 1
#   ./run_keras_train.sh --folds 5 --fold-idx 0 --batch-size 16 --learning-rate 5e-5
set -e
cd "$(dirname "$0")"

PYTHON=".venv-tf/bin/python"
VENV_DIR=".venv-tf"

if [ ! -x "$PYTHON" ]; then
  echo "Keras virtualenv not found: $PYTHON" >&2
  echo "Keras/TensorFlow must run from .venv-tf. Root PyTorch scripts use .venv." >&2
  exit 1
fi

export LD_LIBRARY_PATH="$(find "$VENV_DIR/lib" -path "*/nvidia/*/lib" -type d | tr '\n' ':')"

echo "=== Python environment: $VENV_DIR ==="
echo "=== GPU status ==="
"$PYTHON" - <<'EOF'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(f"GPU: {gpus if gpus else 'none (CPU)'}")
EOF

echo ""
echo "=== Training Keras 5-input multimodal model ==="
"$PYTHON" -m keras_pipeline.train_tf "$@"
