# Face Recognition Pipeline Module Guide

## Scope

- Own the MobileNetV1 ArcFace architecture adaptation, calibration data sampling, and NPU INT8 TFLite export for facial recognition / feature extraction.
- Preserve the NPU operator contract for downstream access-control runtimes.

## Orient First

- Architecture & weights loader: `models/mobilenet_v1.py`
- Calibration data sampler: `data/calibration.py`
- INT8 graph verifier: `export/verifier.py`
- Export CLI entry point: `export/export_npu.py`
- Execution wrapper: `scripts/face_recognition/run_export.sh`

## Boundary & Architecture Constraints

- **Input Contract**: Fixed batch shape `(1, 112, 112, 3)` RGB, normalized to `[-1.0, 1.0]` via `(x - 127.5) / 128.0`.
- **Output Contract**: `(1, 256)` feature embedding vector.
- **NPU Compiler Constraints**:
  - Fixed batch size of 1.
  - Explicit 3×3 Depthwise Conv (kernel 3×3 on 3×3 feature map -> 1×1×1024).
  - 1×1 Conv projection head to 256 dimensions.
  - Reshape `(256,)` instead of Flatten.
  - Zero floating-point tensors in the exported graph (strictly 31 INT8 operators).
- **Calibration Mandate**:
  - Sample only from fixed `dataset/raw/{train,validation}/live` subject directories.
  - Stratified round-robin sampling across subjects (minimum 300 images, default 500).

## Change Gates

- Checkpoint weights (`model/recognition/*.h5`) and exported artifacts (`model/recognition/tflite/*.tflite`) remain gitignored under `model/`.
- Every export must satisfy `verify_int8_operator_graph()` without fallback.
- Calibration must not sample from spoof classes.

## Verify

```bash
uv run pytest tests/face_recognition
bash -n scripts/face_recognition/run_export.sh
```
