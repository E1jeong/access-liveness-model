#!/usr/bin/env bash
# run_single_all_folds.sh - 단일 입력 모델(crop_rgb 또는 crop_ir)의 Fold 0~4 전체 훈련, 변환, 평가 일괄 수행

set -e
cd "$(dirname "$0")"

# 기본값 설정
MODEL_TYPE=""
EPOCHS=30

# 인자 파싱
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --model-type) MODEL_TYPE="$2"; shift ;;
    --epochs) EPOCHS="$2"; shift ;;
    *) echo "알 수 없는 인자: $1"; exit 1 ;;
  esac
  shift
done

if [[ "$MODEL_TYPE" != "crop_rgb" && "$MODEL_TYPE" != "crop_ir" ]]; then
  echo "사용법: $0 --model-type [crop_rgb|crop_ir] [--epochs 30]"
  exit 1
fi

echo "========================================="
echo "  [Start] 5-Fold Training & Conversion"
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
    "model/keras/best_${MODEL_TYPE}_fold${idx}_float.tflite" \
    "model/keras/best_${MODEL_TYPE}_fold${idx}_int8.tflite" \
    "model/keras/best_${MODEL_TYPE}_fold${idx}_npu_int8.tflite"
done

echo ""
echo "========================================="
echo "  [Success] All tasks completed!"
echo "========================================="
