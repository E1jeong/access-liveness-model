#!/usr/bin/env bash
# 고정 train/validation/test 기준 학습 + 변환 + validation 평가.
# test split은 설정 확정 후 evaluate_tflite.py --split test로 별도 실행한다.
set -e
cd "$(dirname "$0")/../.."

MODEL_TYPE="dual"
BACKBONE="mobilenetv2"
EPOCHS=10
BATCH_SIZE=8
LEARNING_RATE="1e-4"
DATA_DIR="dataset/raw"
CALIBRATION_SAMPLES=500
FORCE=""
REDUCTION="sum"
TRAIN_EXTRA=()

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --model-type) MODEL_TYPE="$2"; shift ;;
    --backbone) BACKBONE="$2"; shift ;;
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

case "$BACKBONE" in
  mobilenetv2|efficientnet_lite0|mobilefacenet) ;;
  *) echo "사용법: $0 --backbone [mobilenetv2|efficientnet_lite0|mobilefacenet]"; exit 1 ;;
esac

if [[ "$BACKBONE" == "mobilefacenet" ]]; then
  if [[ "$MODEL_TYPE" != "crop_ir" ]]; then
    echo "MobileFaceNet은 --model-type crop_ir만 지원합니다"; exit 1
  fi
  PREFIX="best_crop_ir_mobilefacenet_fixed"
  TRAIN_EXTRA=(--rgb-weights none)
elif [[ "$BACKBONE" != "mobilenetv2" ]]; then
  if [[ "$MODEL_TYPE" == "dual" ]]; then PREFIX="best_model_${BACKBONE}_fixed"; else PREFIX="best_${MODEL_TYPE}_${BACKBONE}_fixed"; fi
fi

echo "========================================="
echo "  [시작] 고정 split 학습 파이프라인"
echo "  모델 종류      : $MODEL_TYPE"
echo "  백본           : $BACKBONE"
echo "  데이터 경로    : $DATA_DIR"
echo "  에폭            : $EPOCHS"
echo "  배치 크기       : $BATCH_SIZE"
echo "  학습률          : $LEARNING_RATE"
echo "  강제 덮어쓰기   : ${FORCE:-사용 안 함}"
echo "========================================="

.venv-tf/bin/python validate_fixed_splits.py --data-dir "$DATA_DIR"
./scripts/keras/run_keras_train.sh \
  --data-dir "$DATA_DIR" \
  --model-type "$MODEL_TYPE" \
  --backbone "$BACKBONE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --conv1-reduction "$REDUCTION" \
  "${TRAIN_EXTRA[@]}" \
  $FORCE
./scripts/keras/run_keras_convert.sh \
  --data-dir "$DATA_DIR" \
  --model-type "$MODEL_TYPE" \
  --backbone "$BACKBONE" \
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
echo "  [성공] 고정 split 파이프라인 완료"
echo "  test split은 자동 평가하지 않았습니다."
echo "========================================="
