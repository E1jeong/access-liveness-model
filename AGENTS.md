# access-liveness-model AI Guide

## Context

- Governs code navigation, implementation boundaries, and safety for `access-liveness-model`.
- **Project wiki (company-maintained)**: The `ubio-vision-model` folder of the company GitLab repository `ai/ubio-vision/wiki` is the single source of truth for project context, dataset standards, training pipelines, quantization benchmarks, and deployment contracts. Use a pulled checkout of that repository (it may be embedded in a host Vault as a submodule), never a hardcoded file URL; follow its root `AGENTS.md` and update owning pages through reviewed commits or merge requests.
- **Without the wiki**: Code and these `AGENTS.md` guides are authoritative; an unpulled wiki snapshot is read-only background that may be stale and yields to code. Report results with run ID, split, artifact, execution machine, code revision, and metrics in commit messages or merge requests for wiki intake. Wiki page names cited in guides (e.g. `technical/training-pipeline`) are relative to the project folder or snapshot root.
- **Paired Project**: Governs training and quantization upstream for `android-anti-spoofing-lab` (Anti-Spoofing Viewer Android app). Output contracts follow `technical/android-deployment-agreement`.
- **Machine Topology**:
  - **Company PC (WSL CPU)**: Code/doc editing, Git operations, and fast unit tests (`uv run pytest tests/dataset tests/metrics`). Never run training here.
  - **Sub GPU machine (`sub`, GTX 1660 Ti)**: The project owner's training, dataset processing, and INT8 quantization host via managed `uv` virtual environments (`.venv` for PyTorch, `.venv-tf` for Keras TF 2.21).
  - **Mac Studio (`mac`, Apple Silicon M4 Max)**: The collaborator's training host; high-speed Metal GPU training & export node. Uses a unified `.venv-tf` environment (TensorFlow 2.18.0 + `tensorflow-metal 1.2.0` + LiteRT 2.3.0, `requirements/mac-tf.lock`): supports Metal GPU-accelerated training, Keras 3 TFLite export, INT8 quantization, and validation evaluation in a single environment. Grappler meta-optimizer is disabled on Darwin to guarantee numerical parity with TFLite. Holds full 12-class condition-folder `dataset/raw` synchronized with `sub`.
  - **Host sharing**: Each person trains on their own host. Coordinate before running on the other host (one GPU each), and never modify the other person's checkout, run directories, or tmux sessions. Record the execution machine with metrics; `sub` (CUDA, TF 2.21) and `mac` (Metal, TF 2.18) results are compared only with that noted.
- Read the nearest module `AGENTS.md` before changing a pipeline module; this root guide remains in force everywhere.

## Code Map

| Module | Responsibility | First entry point | Module guide |
| --- | --- | --- | --- |
| `keras_pipeline/` | Production training, backbones, NPU INT8 export | `keras_pipeline/training/train.py` | `keras_pipeline/AGENTS.md` |
| `pytorch_pipeline/` | Research backbones, Sony MCT PTQ bridge | `pytorch_pipeline/train.py` | `pytorch_pipeline/AGENTS.md` |
| `face_detection_pipeline/` | Box-only YuNet-N face detector training, evaluation, and TFLite export | `face_detection_pipeline/models/yunet.py` | `face_detection_pipeline/AGENTS.md` |
| `scripts/` | Shell wrappers, CUDA/cuDNN environment preambles | `scripts/keras/_keras_env.sh` | `scripts/AGENTS.md` |
| `tests/` | Automated test suite, leakage checks, export regression | `tests/dataset/test_fixed_splits.py` | `tests/AGENTS.md` |
| Common Tools | Shared 12-class SSOT, leakage detection, evaluation | `common/classes.py`, `common/utils.py`, `common/evaluate_tflite.py` | Root `AGENTS.md` |

## Change Gates

- **Fixed Split Mandate**: All training uses `dataset/raw/{train,validation}` (`test` is optional and leakage-checked when present; 1-per-class placeholder `test` evaluation is blocked). Never reintroduce K-Fold splitting. Run `python -m common.validate_fixed_splits` before training.
- **Conv1 Reduction**: Single-channel IR ImageNet transfer must use `sum` reduction. `mean` is rejected and must never be passed to published runs.
- **Class Names SSOT**: `common/classes.py:CLASS_NAMES` is the sole source of truth for the 12-class order.
- **Shell Wrapper Mandate**: Never invoke bare `python` for Keras on the GPU server. Always run through `scripts/keras/*.sh` so `_keras_env.sh` sets `LD_LIBRARY_PATH` for `libcudnn.so.9`.
- **NNAPI No-Fallback Policy**: Android runtime rejects a model slot on NNAPI setup/warmup failure; it must not fall back to CPU. Ensure all exported TFLite models conform strictly to NPU operators.
- **Android Export Contract**: The downstream Android SDK deploys only `crop_ir` as IR `single_1_input` (`ir@0`, one channel) or `dual` as RGB+IR `dual_2_input` (distinct `rgb`/`ir` indices `0`/`1`, three/one channels), both with one `[1,12]` output. `crop_rgb` may remain a training experiment but is not a standalone Android deployment form; paired one-input and five-input artifacts are historical only.
- **Data & Run Protection**: Never modify or overwrite original capture images (`RGB.bmp`/`IR.bmp`). Store each training run in a separate directory containing its run ID; do not overwrite existing candidate checkpoints (`.pth`, `.keras`, or `.tflite`).
- **Artifact Separation**: Store generated `.tflite`, `.keras`, and `.pth` only in gitignored `model/`. Never commit model weights or raw image datasets to Git.

## Verify

- Company PC fast unit tests: `uv run pytest tests/dataset tests/metrics`
- Split integrity & leakage check: `.venv/bin/python -m common.validate_fixed_splits` (or `.venv-tf/bin/python` on GPU server/mac)
- GPU server Keras export suite: `.venv-tf/bin/pytest tests/export --ignore=tests/export/test_pytorch_mct_export.py`
- GPU server mixed PyTorch MCT + TensorFlow export suite: `.venv/bin/pytest tests/export`
- Keras model & GPU smoke test: `./scripts/keras/run_keras_model.sh`
- GPU-server TFLite evaluation on validation split: `.venv-tf/bin/python -m common.evaluate_tflite --models model/keras/best_crop_ir_fixed_npu_int8.tflite --model-type crop_ir --split validation`
- Report exact commands and results.
