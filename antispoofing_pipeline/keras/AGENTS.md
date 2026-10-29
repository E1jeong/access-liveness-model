# `antispoofing_pipeline.keras` Module Guide

## Scope

- Own production training, validation, evaluation, and TFLite INT8 quantization for the `access-liveness-model` project.
- Implement dataset loading (`data/dataset.py`), pseudo 3D depth generation (`data/depth_generator.py`), model architectures (`models/model.py`, `models/efficientnet_lite.py`), training loops with ACER checkpointing (`training/train.py`), TFLite conversion (`export/converter.py`), NPU export reconstruction (`export/npu_model.py`), and sidecar manifest validation (`export/validator.py`).

## Orient First

- Source entry points:
  - Model definitions: `models/model.py` (`build_single_model`, `build_dual_model`, `extract_deploy_model`), `models/efficientnet_lite.py`
  - 3D Depth supervision: `data/depth_generator.py` (`generate_pseudo_depth_map`, `_build_base_templates`, `generate_high_frequency_residual_map`)
  - Losses: `models/losses.py` (CrossEntropy, Focal Loss, binary PAD, and SupCon)
  - Training loop: `training/train.py` (cosine decay, `AcerCheckpoint`, fixed-split evaluation, auxiliary heads, `--loss focal`)
  - Quantization & export: `export/converter.py` (Full INT8 conversion), `export/npu_model.py` (NPU graph reconstruction and parity)
  - Graph inspection & sidecars: `export/validator.py` (`inspect_tflite`, `write_tflite_sidecar_manifest`)

## Boundary & Architecture Constraints

1. **Fixed Splits Only**: Training and calibration use `dataset/raw/train` and model selection uses `dataset/raw/validation` (`dataset/raw/test` is optional for prospective holdout checks and rejects placeholder evaluation). Never reintroduce K-Fold splitting.
2. **Supported Backbones**:
   - `mobilenetv2`: Baseline architecture; supports `crop_ir` (1ch), `crop_rgb` (3ch), and `dual` (separate RGB 3ch + IR 1ch inputs).
   - `efficientnet_lite0`: Active candidate; supports `crop_ir`, `crop_rgb`, and `dual`.
3. **IR Conv1 Reduction**: Single-channel IR transfer from ImageNet weights must use fixed `sum` reduction; no reduction selector is exposed.
4. **NPU-Friendly INT8 Graph Rules**:
   - Graph input must accept `[-1, 1]` normalized tensors (Lambda input normalization removed from graph).
   - `MEAN` global pooling replaced with `AVERAGE_POOL_2D`.
   - Classifier head built with `1x1 Conv2D` instead of `Dense` to avoid NPU dimension flattening defects.
   - Batch dimension fixed to `1`.
5. **Deterministic Augmentation**: Stream-global `(epoch * N + step, seed)` ensures epoch-unique augmentation without breaking reproducibility.

## Change Gates

- For requested training/export execution or changes to approved presets and deployment contracts, consult the relevant wiki runbook/contract when available; do not treat module navigation as authorization to launch training.
- Always execute via `scripts/keras/*.sh` from repository root. Never invoke bare `python` on the GPU machine (`sub`) because `.venv-tf` requires `libcudnn.so.9` from `_keras_env.sh`. On macOS (`mac`), training must execute on CPU: explicitly hide TensorFlow GPUs before device initialization and verify no GPU is visible; `CUDA_VISIBLE_DEVICES=""` alone does not hide Metal.
- Do not overwrite existing candidate models (`model/keras/best_*`). Write new runs with unique timestamps or run IDs.
- Any change to model inputs, outputs, or quantization must generate and validate the matching sidecar JSON (`best_*_manifest.json`) via `export/validator.py`.

## Verify

```bash
# Model architecture & GPU smoke test
./scripts/keras/run_keras_model.sh

# Unit tests for Keras models and exports
.venv-tf/bin/pytest tests/model/test_conv1_reduction.py tests/model/test_efficientnet_lite.py tests/export/test_npu_export.py
```
