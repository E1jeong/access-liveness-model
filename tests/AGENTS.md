# `tests` Guide (Automated Test Suite & Quality Gates)

## Scope

- Own the automated test suite covering dataset integrity, model architectures, quantization and artifact regression, and evaluation metric computation.

## Orient First

- Read `tests/evaluation-metrics-results` and `schema` (§Information precedence) before adding or updating tests.
- Configuration: `pytest.ini`
- Test directories:
  - Dataset & leakage: `tests/dataset/` (`test_dataset_pipeline.py`, `test_deterministic_augmentation.py`, `test_fixed_splits.py`)
  - Export & quantization: `tests/export/` (`test_artifact_paths.py`, `test_artifact_regression.py`, `test_calibration_sampling.py`, `test_npu_export.py`, `test_pytorch_mct_export.py`)
  - Metrics: `tests/metrics/` (`test_metrics.py`)
  - Model architectures: `tests/model/` (`test_conv1_reduction.py`, `test_efficientnet_lite.py`, `test_mobilefacenet.py`, `test_model_signature.py`, `test_pytorch_model.py`, `test_run_metadata.py`, `test_training_enhancements.py`)

## Boundary & Architecture Constraints

1. **Environment Scope**:
   - Fast unit tests without heavy framework imports can run on CPU (Company PC or GPU server).
   - TensorFlow/Keras and INT8 export tests require `.venv-tf` on the GPU server.
2. **Leakage & Determinism Protection**:
   - Any modification to `utils.py` or `keras_pipeline/tf_dataset.py` must pass `test_fixed_splits.py` and `test_deterministic_augmentation.py`.
   - Never weaken the 4-tier leakage detection rules (subject ID, canonical realpath, MD5 content hash, and session/video metadata).
3. **Artifact Regression**:
   - `test_artifact_regression.py` ensures generated TFLite models conform to output shape `[1, 10]`, valid tensor names, and expected quantization scales.

## Change Gates

- All unit tests must pass before approving architectural or pipeline changes.
- Fast tests must run cleanly on the company PC (CPU environment); full TensorFlow/export tests run on `sub` (GPU environment) in `.venv-tf`.

## Verify

```bash
# Fast unit tests (Company PC or GPU server)
pytest tests/dataset tests/metrics

# Full test suite (GPU server with .venv-tf)
pytest
```
