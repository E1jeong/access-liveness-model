#!/usr/bin/env bash
# pytorch_pipeline/convert_to_tflite.py — PyTorch 모델 -> TFLite 변환 실행기
#
# 사용 예:
#   ./scripts/pytorch/run_pytorch_convert.sh                          # 기본 경로 모델 변환
#   ./scripts/pytorch/run_pytorch_convert.sh --pth-path model/best_model_fold1.pth --tflite-path model/anti_spoofing.tflite
set -e
cd "$(dirname "$0")/../.."

echo "=== TFLite 변환 시작 ==="
.venv/bin/python -m pytorch_pipeline.convert_to_tflite "$@"
