# `tests` Module Guide

## Scope

- Own the automated test suite covering dataset integrity, model architectures, quantization and artifact regression, and evaluation metric computation.

## Orient First

- Configuration: `pytest.ini`
- Test directories:
  - Dataset & leakage: `tests/dataset/` (`test_dataset_pipeline.py`, `test_deterministic_augmentation.py`, `test_fixed_splits.py`)
  - Export & quantization: `tests/export/` (`test_artifact_paths.py`, `test_artifact_regression.py`, `test_calibration_sampling.py`, `test_evaluate_keras_model.py`, `test_evaluate_npu_int8_contract.py`, `test_npu_export.py`, `test_pytorch_mct_export.py`, `test_recognition_calibration.py`, `test_sidecar_manifest.py`)
  - Metrics: `tests/metrics/` (`test_metrics.py`)
  - Model architectures & loss: `tests/model/` (`test_aux_depth.py`, `test_aux_residual.py`, `test_binary_pad.py`, `test_conv1_reduction.py`, `test_display_target_augmentation.py`, `test_efficientnet_lite.py`, `test_focal_loss.py`, `test_model_signature.py`, `test_pytorch_model.py`, `test_run_metadata.py`, `test_supcon.py`, `test_training_enhancements.py`)
  - Face detection: `tests/face_detection/` (model, adapter, loss parity, training, decoding, metrics, and export contracts)

## Boundary & Architecture Constraints

1. **Environment Scope**:
   - Fast unit tests without heavy framework imports can run on CPU (Company PC or GPU server).
   - TensorFlow/Keras-only export tests use `.venv-tf` on the GPU server.
   - The mixed PyTorch MCT + TensorFlow export suite uses `.venv` on `sub`, where both stacks are installed.
2. **Leakage & Determinism Protection**:
   - Any modification to `common/utils.py` or `keras_pipeline/data/dataset.py` must pass `test_fixed_splits.py` and `test_deterministic_augmentation.py`.
   - Never weaken evidence-based leakage detection: canonical realpath, MD5 content hash, and session/video metadata. Folder numbers are split-local capture IDs, not identity evidence; add identity leakage checks only when an explicit metadata `groupId` contract exists.
3. **Artifact Contracts**:
   - `test_model_signature.py` enforces the 12-class order and Keras/TFLite input names, channels, and `[1, 12]` output.
   - `test_artifact_regression.py` covers `common/evaluate_tflite.py` regression reports (logit metrics, size, latency, label alignment) and explicit-split metric CSVs.

## Change Gates

- Report model metrics with run ID, split, artifact, execution machine, and code revision. Saved-dataset evaluation does not verify Android behavior.
- All unit tests must pass before approving architectural or pipeline changes.
- Fast tests must run cleanly on the company PC (CPU environment); Keras-only export tests run in `.venv-tf`, while the mixed PyTorch MCT + TensorFlow export suite runs in `.venv` on `sub`.

## Verify

```bash
# Fast unit tests (Company PC or GPU server)
uv run pytest tests/dataset tests/metrics

# Keras export suite (GPU server or mac with .venv-tf)
.venv-tf/bin/pytest tests/export --ignore=tests/export/test_pytorch_mct_export.py

# Mixed PyTorch MCT + TensorFlow export suite (sub .venv)
.venv/bin/pytest tests/export
```
