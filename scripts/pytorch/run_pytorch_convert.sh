#!/usr/bin/env bash
# pytorch_pipeline/convert_to_tflite.py — PyTorch 모델 -> Sony MCT -> TFLite 변환 실행기
#
# 사용 예:
#   ./scripts/pytorch/run_pytorch_convert.sh --pth-path model/pytorch/best_crop_ir_mobilenetv3_fixed.pth --output-prefix model/pytorch/best_crop_ir_mobilenetv3_fixed --model-type crop_ir
set -e
cd "$(dirname "$0")/../.."

echo "=== PyTorch Sony MCT TFLite 변환 시작 ==="
.venv/bin/python -m pytorch_pipeline.convert_to_tflite "$@"
