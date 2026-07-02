#!/usr/bin/env bash
# Convert a saved Keras 5-input multimodal model to TFLite.
#
# Examples:
#   ./run_keras_convert.sh --float --int8
#   ./run_keras_convert.sh --float --int8 --fold-idx 1
#   ./run_keras_convert.sh --npu-int8 --fold-idx 4 --calibration-samples 500
set -e
cd "$(dirname "$0")"
source ./keras_env.sh

echo ""
echo "=== Converting Keras 5-input multimodal model ==="
"$PYTHON" -m keras_pipeline.convert_h5_to_tflite "$@"
