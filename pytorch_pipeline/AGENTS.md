# `pytorch_pipeline` Guide (R&D Sandbox & Quantization Bridge)

## Scope

- Own research, prototyping, and PyTorch-based model architectures (`SingleInputMobileNetV3`, `DualInputMobileNetV3`).
- Host the Sony Model Compression Toolkit (MCT) bridge for post-training quantization (PTQ) and quantization-aware training (QAT) to emit Android NPU-compliant TFLite INT8 models.

## Orient First

- Read `technical/training-pipeline` (§65–90), `technical/int8-quantization-npu`, and `technical/training-command-guide` (§5) before modifying PyTorch models or Sony MCT quantization.
- Source entry points:
  - Model definitions: `model.py` (`SingleInputMobileNetV3`, `DualInputMobileNetV3`, `replace_hardswish_with_relu`, `disable_se_blocks`)
  - Training loop: `train.py`
  - Sony MCT / TFLite export: `convert_to_tflite.py` (PTQ/QAT via ONNX -> onnx2tf)
  - Verification & setup: `verify_setup.py`, `verify_quantization.py`, `verify_raw_pth.py`

## Boundary & Architecture Constraints

1. **NPU-Friendly Operator Constraints**:
   - `Hardswish` must be replaced with `ReLU6` (`replace_hardswish_with_relu`).
   - Squeeze-and-Excitation (`SE`) blocks must be disabled (`disable_se_blocks`).
   - Classifier head must use `_build_npu_classifier_head` (`1×1 Conv2D` -> `ReLU6` -> `Dropout` -> `1×1 Conv2D`) instead of `torch.flatten` + `nn.Linear`.
   - Conv1 reduction for IR single-channel must be `sum`.
2. **Sony MCT Quantization Bridge**:
   - Standard PyTorch PTQ causes activation collapse on MobileNetV3; use the Sony MCT path with QAT fine-tuning for INT8 convergence.
3. **Android Contract Compatibility**:
   - The emitted TFLite model from `convert_to_tflite.py` (via ONNX -> `onnx2tf`) must strictly follow the Android contract: NHWC input `[1, 224, 224, 1]` or `[1, 224, 224, 3]`, output `[1, 12]` INT8 logits (matching `len(CLASS_NAMES)`).
   - Sidecar manifest must be generated and inspected using `keras_pipeline.export_validator`.

## Change Gates

- Always use `.venv` (Python 3.12, PyTorch cu128).
- Do not introduce operations unsupported by VeriSilicon VIP8000 NPU (avoid arbitrary non-linearities, dynamic shapes, or per-channel dequantize ONNX ops).
- Do not deploy PyTorch-origin INT8 models without full verification against `evaluate_tflite.py`.

## Verify

```bash
# Verify environment and CUDA / litert_torch / MCT setup
./scripts/pytorch/run_pytorch_verify.sh

# Model shape and architecture smoke test
./scripts/pytorch/run_pytorch_model.sh

# End-to-end PyTorch fixed split run
./scripts/pytorch/run_fixed_split.sh --model-type crop_ir --epochs 30 --batch-size 32 --learning-rate 2e-4

# Unit tests for PyTorch models and MCT export
pytest tests/model/test_pytorch_model.py tests/export/test_pytorch_mct_export.py
```
