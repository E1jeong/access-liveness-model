#!/usr/bin/env bash
# Print the Keras multimodal model summary without training.
set -e
cd "$(dirname "$0")"
source ./keras_env.sh

echo ""
echo "=== MobileNetV2 5-input multimodal model summary ==="
"$PYTHON" -m keras_pipeline.tf_model "$@"
