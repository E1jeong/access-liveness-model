# `scripts` Guide (Execution Wrappers & Server Tooling)

## Scope

- Own shell execution wrappers for training, conversion, verification, and git synchronization across working environments.
- Provide environment preambles for CUDA/cuDNN dynamic library paths.

## Orient First

- Read `technical/training-command-guide` and `operations/working-environment` before running server scripts or adding wrappers.
- Source entry points:
  - Keras environment preamble: `scripts/keras/_keras_env.sh`
  - Keras fixed-split wrapper: `scripts/keras/run_fixed_split.sh`
  - PyTorch fixed-split wrapper: `scripts/pytorch/run_fixed_split.sh`
  - Recognition NPU export: `scripts/recognition/export_npu_mobilenet_emore.py`
  - Face detector training: `scripts/face_detection/run_train.sh`
  - Git synchronization cleaner: `scripts/git_pull_clean.sh`

## Boundary & Architecture Constraints

1. **Root-Relative Execution**: All scripts must resolve the repository root dynamically (`cd "$(dirname "$0")/../.."`) and be callable from the repository root.
2. **cuDNN Library Resolution**: Keras scripts must source `scripts/keras/_keras_env.sh` to populate `LD_LIBRARY_PATH` from `.venv-tf/lib/.../nvidia/*/lib`. Bare Python execution fails to locate GPU cuDNN libraries on the training host.
3. **CRLF Prevention & Line Endings**: All scripts must maintain strict LF line endings, enforced by `.gitattributes` (`*.sh text eol=lf`). Executable shell wrappers are tracked as `+x`; source-only `_keras_env.sh` is not.
4. **Git Pull Hygiene**: On the GPU server (`sub`), always synchronize code using `scripts/git_pull_clean.sh`. This stashes, pulls with rebase, and avoids deleting gitignored `dataset/`, `model/`, or virtual environments.
5. **Background Execution**: Use `sub-train` on the GPU server (or `mac-train` on Mac Studio) so long training runs survive SSH disconnection.

## Change Gates

- Test shell script syntax before committing: `bash -n scripts/**/*.sh`.
- Never hardcode host-specific absolute paths (e.g., `/home/union/...` or `C:\Users\...`) inside scripts.
- Default arguments in scripts must match repository-wide constants (`REDUCTION="sum"`, `DATA_DIR="dataset/raw"`; `CALIBRATION_SAMPLES=500` for Keras and `200` for PyTorch, matching `convert_to_tflite.py --calib-samples`).

## Verify

```bash
# Syntax verification
bash -n scripts/keras/*.sh
bash -n scripts/pytorch/*.sh
bash -n scripts/face_detection/*.sh
bash -n scripts/git_pull_clean.sh

# GPU environment check
./scripts/keras/run_keras_model.sh
./scripts/pytorch/run_pytorch_verify.sh
```
