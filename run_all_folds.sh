#!/usr/bin/env bash
# run_all_folds.sh - Fold 0~4 훈련, TFLite 변환(float/int8/npu-int8), 전체 fold 평가 일괄 수행

set -e
cd "$(dirname "$0")"

echo "========================================="
echo "  [Start] Fold 0~4 Training & Conversion"
echo "========================================="
for idx in 0 1 2 3 4; do
  echo ">>> Processing Fold $idx / 4"
  ./run_keras_train.sh --epochs 10 --folds 5 --fold-idx $idx
  ./run_keras_convert.sh --float --int8 --npu-int8 --folds 5 --fold-idx $idx
done

echo "========================================="
echo "  [Start] Evaluation for All Folds (0~4)"
echo "========================================="
# evaluate_tflite.py는 tensorflow를 import하므로 반드시 .venv-tf로 실행한다.
# 한 fold의 세 모델은 단일 데이터셋 패스로 함께 평가된다.
for idx in 0 1 2 3 4; do
  echo ">>> Evaluating Fold $idx"
  .venv-tf/bin/python evaluate_tflite.py --folds 5 --fold-idx $idx --models \
    model/keras/best_model_fold${idx}_float.tflite \
    model/keras/best_model_fold${idx}_int8.tflite \
    model/keras/best_model_fold${idx}_npu_int8.tflite
done

echo "========================================="
echo "  [Success] All tasks completed!"
echo "========================================="
