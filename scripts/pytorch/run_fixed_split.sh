#!/usr/bin/env bash
# PyTorch 고정 train/validation/test 기준 학습 + Sony MCT 변환 + validation 평가.
# test split은 설정 확정 후 evaluate_tflite.py --split test로 별도 실행한다.
set -e
cd "$(dirname "$0")/../.."

MODEL_TYPE="crop_ir"
EPOCHS=10
BATCH_SIZE=8
LEARNING_RATE="1e-4"
DATA_DIR="dataset/raw"
CALIBRATION_SAMPLES=200
REDUCTION="sum"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --model-type) MODEL_TYPE="$2"; shift ;;
    --epochs) EPOCHS="$2"; shift ;;
    --batch-size) BATCH_SIZE="$2"; shift ;;
    --learning-rate) LEARNING_RATE="$2"; shift ;;
    --data-dir) DATA_DIR="$2"; shift ;;
    --calibration-samples) CALIBRATION_SAMPLES="$2"; shift ;;
    --conv1-reduction) REDUCTION="$2"; shift ;;
    *) echo "알 수 없는 인자: $1"; exit 1 ;;
  esac
  shift
done

case "$MODEL_TYPE" in
  dual) PREFIX="best_dual_mobilenetv3_fixed" ;;
  crop_rgb|crop_ir) PREFIX="best_${MODEL_TYPE}_mobilenetv3_fixed" ;;
  *) echo "사용법: $0 --model-type [dual|crop_rgb|crop_ir] [--epochs N] [--batch-size B] [--learning-rate L] [--data-dir PATH]"; exit 1 ;;
esac

echo "========================================="
echo "  [시작] PyTorch 고정 split 학습 및 Sony MCT 파이프라인"
echo "  모델 종류      : $MODEL_TYPE"
echo "  백본           : mobilenetv3_small (NPU-friendly)"
echo "  데이터 경로    : $DATA_DIR"
echo "  에폭            : $EPOCHS"
echo "  배치 크기       : $BATCH_SIZE"
echo "  학습률          : $LEARNING_RATE"
echo "  Conv1 축소      : $REDUCTION"
echo "  캘리브레이션 수 : $CALIBRATION_SAMPLES"
echo "========================================="

.venv/bin/python validate_fixed_splits.py --data-dir "$DATA_DIR"

./scripts/pytorch/run_pytorch_train.sh \
  --data-dir "$DATA_DIR" \
  --model-type "$MODEL_TYPE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --conv1-reduction "$REDUCTION" \
  --output-dir "model/pytorch" \
  --save-name "${PREFIX}.pth"

./scripts/pytorch/run_pytorch_convert.sh \
  --pth-path "model/pytorch/${PREFIX}.pth" \
  --output-prefix "model/pytorch/${PREFIX}" \
  --model-type "$MODEL_TYPE" \
  --calib-samples "$CALIBRATION_SAMPLES" \
  --dataset-dir "$DATA_DIR/train"

.venv/bin/python evaluate_tflite.py \
  --data-dir "$DATA_DIR" \
  --split validation \
  --model-type "$MODEL_TYPE" \
  --models \
    "model/pytorch/${PREFIX}_float.tflite" \
    "model/pytorch/${PREFIX}_int8.tflite" \
    "model/pytorch/${PREFIX}_npu_int8.tflite"

echo "========================================="
echo "  [성공] PyTorch 고정 split 파이프라인 완료"
echo "  test split은 자동 평가하지 않았습니다."
echo "========================================="
