#!/usr/bin/env bash
# Common environment setup for run_keras_*.sh — source this file, do not execute.
# Sets $PYTHON to the .venv-tf interpreter and exports LD_LIBRARY_PATH so
# TensorFlow finds the pip-installed CUDA libraries (see docs/project_status.md §5).

PYTHON=".venv-tf/bin/python"
VENV_DIR=".venv-tf"

if [ ! -x "$PYTHON" ]; then
  echo "Keras virtualenv not found: $PYTHON" >&2
  echo "Keras/TensorFlow must run from .venv-tf. Root PyTorch scripts use .venv." >&2
  exit 1
fi

export LD_LIBRARY_PATH="$(find "$VENV_DIR/lib" -path "*/nvidia/*/lib" -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"

echo "=== Python environment: $VENV_DIR ==="
echo "=== GPU status ==="
"$PYTHON" - <<'EOF'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(f"GPU: {gpus if gpus else 'none (CPU)'}")
EOF
