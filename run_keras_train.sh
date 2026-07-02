#!/usr/bin/env bash
# Train the Keras 5-input multimodal MobileNetV2 pipeline.
#
# Examples:
#   ./run_keras_train.sh
#   ./run_keras_train.sh --epochs 30 --fold-idx 1
#   ./run_keras_train.sh --folds 5 --fold-idx 0 --batch-size 16 --learning-rate 5e-5
set -e
cd "$(dirname "$0")"
source ./keras_env.sh

echo ""
echo "=== Training Keras 5-input multimodal model ==="
"$PYTHON" -m keras_pipeline.train_tf "$@"
