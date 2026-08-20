# `keras_pipeline` Guide (Production Mainline)

## Scope

- Own production training, validation, evaluation, and TFLite INT8 quantization for the `access-liveness-model` project.
- Implement dataset loading (`tf_dataset.py`), model architectures (`tf_model.py`, `mobilefacenet.py`, `efficientnet_lite.py`), training loops with ACER checkpointing (`tf_train.py`), TFLite export (`convert_keras_to_tflite.py`), and sidecar manifest validation (`export_validator.py`).

## Orient First

- Read `technical/training-pipeline`, `technical/training-command-guide`, `technical/int8-quantization-npu`, and `technical/android-deployment-agreement` before modifying model architectures or quantization pipelines.
- Source entry points:
  - Model definitions: `tf_model.py` (`build_single_input_mobilenetv2`, `build_dual_input_mobilenetv2`), `mobilefacenet.py`, `efficientnet_lite.py`
  - Training loop: `tf_train.py` (cosine decay, `AcerCheckpoint`, fixed-split evaluation)
  - Quantization & export: `convert_keras_to_tflite.py` (Full INT8 and NPU-friendly INT8)
  - Graph inspection & sidecars: `export_validator.py` (`inspect_tflite_graph`, `generate_sidecar_manifest`)

## Boundary & Architecture Constraints

1. **Fixed Splits Only**: Training and calibration use `dataset/raw/train`, model selection uses `dataset/raw/validation`, and final evaluation uses `dataset/raw/test`. Never reintroduce K-Fold splitting.
2. **Supported Backbones**:
   - `mobilenetv2`: Baseline architecture; supports `crop_ir` (1ch), `crop_rgb` (3ch), and `dual` (4ch).
   - `efficientnet_lite0`: Active candidate; supports `crop_ir`, `crop_rgb`, and `dual`.
   - `mobilefacenet`: Scratch-only IR candidate; rejects `crop_rgb` and `dual`; uses ReLU6 and 7×7 GDConv.
3. **IR Conv1 Reduction**: Single-channel IR transfer from ImageNet weights must use `sum` reduction. `mean` is the rejected variant and must never be passed to published runs.
4. **NPU-Friendly INT8 Graph Rules**:
   - Graph input must accept `[-1, 1]` normalized tensors (Lambda input normalization removed from graph).
   - `MEAN` global pooling replaced with `AVERAGE_POOL_2D`.
   - Classifier head built with `1x1 Conv2D` instead of `Dense` to avoid NPU dimension flattening defects.
   - Batch dimension fixed to `1`.
5. **Label Smoothing**: `--label-smoothing 0.1` is standard to prevent logit saturation during quantization.
6. **Deterministic Augmentation**: Stream-global `(epoch * N + step, seed)` ensures epoch-unique augmentation without breaking reproducibility.

## Change Gates

- Always execute via `scripts/keras/*.sh` from repository root. Never invoke bare `python` on the GPU machine (`sub`) because `.venv-tf` requires `libcudnn.so.9` from `_keras_env.sh`.
- Do not overwrite existing candidate models (`model/keras/best_*`). Write new runs with unique timestamps or run IDs.
- Any change to model inputs, outputs, or quantization must generate and validate the matching sidecar JSON (`best_*_manifest.json`) via `export_validator.py`.

## Verify

```bash
# Model architecture & GPU smoke test
./scripts/keras/run_keras_model.sh

# End-to-end training + quantization + validation evaluation
./scripts/keras/run_fixed_split.sh --model-type crop_ir --backbone mobilenetv2 --epochs 30 --batch-size 16 --learning-rate 2e-4

# Unit tests for Keras models and exports
.venv-tf/bin/pytest tests/model/test_conv1_reduction.py tests/model/test_mobilefacenet.py tests/model/test_efficientnet_lite.py tests/export/test_npu_export.py
```
