# `pytorch_pipeline` Module Guide

## Scope

- Own research, prototyping, and PyTorch-based model architectures (`SingleInputMobileNetV3`, `DualInputMobileNetV3`).
- Host the Sony Model Compression Toolkit (MCT) post-training quantization (PTQ) bridge to emit Android NPU-compliant TFLite INT8 models. QAT is not implemented.

## Orient First

- Source entry points:
  - Model definitions: `model.py` (`SingleInputMobileNetV3`, `DualInputMobileNetV3`, `replace_hardswish_with_relu`, `disable_se_blocks`)
  - Training loop: `train.py`
  - Sony MCT / TFLite export: `convert_to_tflite.py` (MCT PTQ -> ONNX -> onnx2tf)
  - Verification & setup: `verify_setup.py`, `verify_quantization.py`

## Boundary & Architecture Constraints

1. **NPU-Friendly Operator Constraints**:
   - `Hardswish` must be replaced with `ReLU6` (`replace_hardswish_with_relu`).
   - Squeeze-and-Excitation (`SE`) blocks must be disabled (`disable_se_blocks`).
   - Classifier head must use `_build_npu_classifier_head` (`1×1 Conv2D` -> `ReLU6` -> `Dropout` -> `1×1 Conv2D`) instead of `torch.flatten` + `nn.Linear`.
   - Conv1 reduction for IR single-channel must be `sum`.
2. **Sony MCT Quantization Bridge**:
   - Standard PyTorch PTQ causes activation collapse on MobileNetV3; use the Sony MCT PTQ path (`get_npu_tflite_tpc`) for INT8 export.
3. **Android Contract Compatibility**:
   - Android deployment accepts only NHWC `[1, 224, 224, 1]` (`crop_ir`) or RGB `[1, 224, 224, 3]` + IR `[1, 224, 224, 1]` (`dual`), with `[1, 12]` INT8 logits matching `len(CLASS_NAMES)`. The converter also supports `crop_rgb` for research; RGB-only export is not an Android-deployable contract.
   - Sidecar manifest must be generated and inspected using `keras_pipeline.export.validator`.

## Change Gates

- Always use `.venv` (Python 3.12, PyTorch cu128).
- Do not introduce operations unsupported by VeriSilicon VIP8000 NPU (avoid arbitrary non-linearities, dynamic shapes, or per-channel dequantize ONNX ops).
- Do not deploy PyTorch-origin INT8 models without full verification against `common/evaluate_tflite.py`.

## Verify

```bash
# Verify environment and CUDA / litert_torch / MCT setup
./scripts/pytorch/run_pytorch_verify.sh

# Model shape and architecture smoke test
./scripts/pytorch/run_pytorch_model.sh

# Unit tests for PyTorch models and MCT export
.venv/bin/pytest tests/model/test_pytorch_model.py tests/export/test_pytorch_mct_export.py
```
