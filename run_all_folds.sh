#!/usr/bin/env bash
# Fold 0~4 전체 학습 + 변환(float/int8/npu-int8) + 평가 일괄 수행.
# dual/multimodal/crop_rgb/crop_ir 네 가지 모델 타입을 모두 지원한다.
#
# 사용 예:
#   ./run_all_folds.sh                                    # dual, 10 에포크 (기본값)
#   ./run_all_folds.sh --model-type multimodal --epochs 30
#   ./run_all_folds.sh --model-type crop_rgb --epochs 30
set -e
cd "$(dirname "$0")"

MODEL_TYPE="dual"
EPOCHS=10

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --model-type) MODEL_TYPE="$2"; shift ;;
    --epochs) EPOCHS="$2"; shift ;;
    *) echo "알 수 없는 인자: $1"; exit 1 ;;
  esac
  shift
done

case "$MODEL_TYPE" in
  dual) PREFIX="best_model" ;;
  multimodal|crop_rgb|crop_ir) PREFIX="best_${MODEL_TYPE}" ;;
  *) echo "사용법: $0 --model-type [dual|multimodal|crop_rgb|crop_ir] [--epochs N]"; exit 1 ;;
esac

echo "========================================="
echo "  [Start] Fold 0~4 Training & Conversion"
echo "  Model Type : $MODEL_TYPE"
echo "  Epochs     : $EPOCHS"
echo "========================================="

for idx in 0 1 2 3 4; do
  echo ""
  echo ">>> Processing Fold $idx / 4"
  ./run_keras_train.sh --model-type "$MODEL_TYPE" --epochs "$EPOCHS" --fold-idx "$idx"
  ./run_keras_convert.sh --model-type "$MODEL_TYPE" --fold-idx "$idx" --float --int8 --npu-int8
done

echo ""
echo "========================================="
echo "  [Start] Evaluation for All Folds (0~4)"
echo "========================================="
for idx in 0 1 2 3 4; do
  echo ""
  echo ">>> Evaluating Fold $idx"
  .venv/bin/python evaluate_tflite.py --model-type "$MODEL_TYPE" --folds 5 --fold-idx "$idx" --models \
    "model/keras/${PREFIX}_fold${idx}_float.tflite" \
    "model/keras/${PREFIX}_fold${idx}_int8.tflite" \
    "model/keras/${PREFIX}_fold${idx}_npu_int8.tflite"
done

echo ""
echo "========================================="
echo "  [Success] All tasks completed!"
echo "========================================="
