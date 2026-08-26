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
  - Git synchronization cleaner: `scripts/git_pull_clean.sh`

## Boundary & Architecture Constraints

1. **Root-Relative Execution**: All scripts must resolve the repository root dynamically (`cd "$(dirname "$0")/../.."`) and be callable from the repository root.
2. **cuDNN Library Resolution**: Keras scripts must source `scripts/keras/_keras_env.sh` to populate `LD_LIBRARY_PATH` from `.venv-tf/lib/.../nvidia/*/lib`. Bare Python execution fails to locate GPU cuDNN libraries on the training host.
3. **CRLF Prevention & Line Endings**: All scripts must maintain strict LF line endings and executable (`+x`) permissions. Git is configured with `core.autocrlf=input`.
4. **Git Pull Hygiene**: On the GPU server (`sub`), always synchronize code using `scripts/git_pull_clean.sh`. This stashes, pulls with rebase, and avoids deleting gitignored `dataset/`, `model/`, or virtual environments.
5. **Background Execution**: Use `sub-train` (or tmux session `train`) on the GPU server so long training runs survive SSH disconnection.

## Change Gates

- Test shell script syntax before committing: `bash -n scripts/**/*.sh`.
- Never hardcode host-specific absolute paths (e.g., `/home/union/...` or `C:\Users\...`) inside scripts.
- Default arguments in scripts must match repository-wide constants (`REDUCTION="sum"`, `DATA_DIR="dataset/raw"`, `CALIBRATION_SAMPLES=500`).

## Verify

```bash
# Syntax verification
bash -n scripts/keras/*.sh
bash -n scripts/pytorch/*.sh
bash -n scripts/git_pull_clean.sh

# GPU environment check
./scripts/keras/run_keras_model.sh
./scripts/pytorch/run_pytorch_verify.sh
```
