#!/usr/bin/env bash
# 고정 train/validation/test 기준 학습 + 변환 + validation 평가.
# test split은 설정 확정 후 evaluate_tflite.py --split test로 별도 실행한다.
set -e
cd "$(dirname "$0")"

MODEL_TYPE="dual"
EPOCHS=10
BATCH_SIZE=8
LEARNING_RATE="1e-4"
DATA_DIR="dataset/raw"
CALIBRATION_SAMPLES=500
FORCE=""
REDUCTION="mean"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --model-type) MODEL_TYPE="$2"; shift ;;
    --epochs) EPOCHS="$2"; shift ;;
    --batch-size) BATCH_SIZE="$2"; shift ;;
    --learning-rate) LEARNING_RATE="$2"; shift ;;
    --data-dir) DATA_DIR="$2"; shift ;;
    --calibration-samples) CALIBRATION_SAMPLES="$2"; shift ;;
    --force) FORCE="--force" ;;
    --conv1-reduction) REDUCTION="$2"; shift ;;
    *) echo "알 수 없는 인자: $1"; exit 1 ;;
  esac
  shift
done

case "$MODEL_TYPE" in
  dual) PREFIX="best_model_fixed" ;;
  crop_rgb|crop_ir) PREFIX="best_${MODEL_TYPE}_fixed" ;;
  *) echo "사용법: $0 --model-type [dual|crop_rgb|crop_ir] [--epochs N] [--batch-size B] [--learning-rate L] [--data-dir PATH] [--force]"; exit 1 ;;
esac

echo "========================================="
echo "  [Start] Fixed Split Training Pipeline"
echo "  Model Type    : $MODEL_TYPE"
echo "  Data Dir      : $DATA_DIR"
echo "  Epochs        : $EPOCHS"
echo "  Batch Size    : $BATCH_SIZE"
echo "  Learning Rate : $LEARNING_RATE"
echo "  Force Overwrite: ${FORCE:-False}"
echo "========================================="

.venv-tf/bin/python validate_fixed_splits.py --data-dir "$DATA_DIR"
./run_keras_train.sh \
  --data-dir "$DATA_DIR" \
  --model-type "$MODEL_TYPE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --conv1-reduction "$REDUCTION" \
  $FORCE
./run_keras_convert.sh \
  --data-dir "$DATA_DIR" \
  --model-type "$MODEL_TYPE" \
  --calibration-samples "$CALIBRATION_SAMPLES" \
  --float --int8 --npu-int8 \
  $FORCE

.venv-tf/bin/python evaluate_tflite.py \
  --data-dir "$DATA_DIR" \
  --split validation \
  --model-type "$MODEL_TYPE" \
  --models \
    "model/keras/${PREFIX}_float.tflite" \
    "model/keras/${PREFIX}_int8.tflite" \
    "model/keras/${PREFIX}_npu_int8.tflite"

echo "========================================="
echo "  [Success] Fixed split pipeline complete"
echo "  test split은 자동 평가하지 않았습니다."
echo "========================================="
