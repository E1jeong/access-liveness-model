# access-liveness-model AI Guide

## Start Here

- This is a navigation aid, not a history archive: help the AI locate the pipeline flow, source entry points, and authoritative knowledge before working.
- The Obsidian wiki `Dev/Project/Company/access-liveness-model` is the source of truth for project context, dataset standards, training pipelines, quantization benchmarks, and deployment contracts.
- Before resuming work or making non-trivial changes, read `README.md` → `handoff.md` → `issues/needs-verification.md`. Identify the working machine with:
  ```bash
  nvidia-smi 2>/dev/null | grep -q "GTX 1660 Ti" && echo "sub-laptop" || echo "company PC"
  ```
- Before proposing or executing commands, explain the command and rationale in Korean. Report the machine, latest completion, next work, and Android/NPU status in Korean.
- Read the nearest `AGENTS.md` before changing a pipeline module.

## Machine Topology and Environment Direction

- **Company PC (WSL CPU)**: Code editing, documentation, Git operations, and fast unit tests (`pytest tests/dataset tests/metrics`). **Never run training here.**
- **Sub-laptop GPU (`sub`, GTX 1660 Ti)**: The sole authoritative environment for training, dataset processing, and INT8 quantization. Uses managed `uv` virtual environments (`.venv` for PyTorch, `.venv-tf` for Keras). Synchronize code from Company PC via Git or `rsync`.

## Product and Pipeline Map

Dual RGB+IR camera anti-spoofing model pipeline targeting the NXP i.MX 8M Plus VeriSilicon NPU on Android terminals (`com.virditech.ac7000`).

```text
RGB + IR Dataset (dataset/raw/{train,validation,test})
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
[ keras_pipeline/ ]             [ pytorch_pipeline/ ]
(Production Mainline)           (R&D Sandbox & Bridge)
 MobileNetV2 / EfficientNet      MobileNetV3 (ReLU6 / no SE)
 Cosine Decay + AcerCheckpoint   Sony MCT PTQ / QAT Bridge
       │                               │
       ▼                               ▼
convert_keras_to_tflite.py      convert_to_tflite.py (ONNX->onnx2tf)
(NPU-friendly INT8 [-1,1])      (NPU-compliant INT8)
       │                               │
       └───────────────┬───────────────┘
                       ▼
         [ model/keras/ or model/pytorch/ ]
          .tflite + sidecar manifest.json
                       │
                       ▼
       evaluate_tflite.py (--split test)
                       │
                       ▼
 Android Target Boundary: android-anti-spoofing-lab
 (app/src/main/assets/ best_*_fixed_npu_int8.tflite)
```

## Module Map and First Reads

| Module | Ownership | First source entry point | Related wiki topics |
| --- | --- | --- | --- |
| `keras_pipeline/` | Production training, backbones, NPU INT8 export | `tf_model.py`, `tf_train.py`, `convert_keras_to_tflite.py` | `technical/training-pipeline`, `technical/int8-quantization-npu` |
| `pytorch_pipeline/` | Research backbones, Sony MCT PTQ/QAT bridge | `model.py`, `train.py`, `convert_to_tflite.py` | `technical/training-pipeline` (§65–90), `technical/training-command-guide` |
| `scripts/` | Shell wrappers, CUDA/cuDNN environment preambles | `scripts/keras/_keras_env.sh`, `scripts/git_pull_clean.sh` | `technical/training-command-guide`, `operations/working-environment` |
| `tests/` | Automated test suite, leakage checks, export regression | `tests/dataset/test_fixed_splits.py`, `pytest.ini` | `tests/evaluation-metrics-results`, `schema` |
| Core Tools | Shared 10-class SSOT, leakage detection, evaluation | `classes.py`, `utils.py`, `evaluate_tflite.py` | `features/classification-system`, `data/dataset-standard` |

## Task Router

| Request concerns | Read first | First source path | Then trace |
| --- | --- | --- | --- |
| Train Keras model candidate | `technical/training-command-guide`, `technical/training-pipeline` | `scripts/keras/run_fixed_split.sh` | `keras_pipeline/tf_train.py` → `tf_model.py` → `tf_dataset.py` |
| INT8 quantization / NPU export | `technical/int8-quantization-npu`, `technical/android-deployment-agreement` | `keras_pipeline/convert_keras_to_tflite.py` | `keras_pipeline/export_validator.py` → generated sidecar JSON |
| Independent test set evaluation | `tests/evaluation-metrics-results`, `data/dataset-standard` | `evaluate_tflite.py` | `utils.py` (metrics calculation) → report table |
| PyTorch training / Sony MCT | `technical/training-pipeline`, `technical/training-command-guide` | `scripts/pytorch/run_fixed_split.sh` | `pytorch_pipeline/convert_to_tflite.py` → `model.py` |
| Dataset validation & split integrity | `data/dataset-standard`, `features/classification-system` | `validate_fixed_splits.py` | `utils.py` (4-tier leakage detection) |
| Android deployment / sidecar manifest | `technical/android-deployment-agreement` | `keras_pipeline/export_validator.py` | `android-anti-spoofing-lab: AntiSpoofingClassifier.java` |
| Concept comprehension review | `operations/working-environment` | `docs/keras-concept-review.md` | User-driven review only (see Change Gates) |

## Immutable Project Boundaries and Change Gates

1. **Fixed Split Mandate**: All future training uses `dataset/raw/{train,validation,test}`. Never reintroduce K-Fold splitting. Run `validate_fixed_splits.py` before training.
2. **Conv1 Reduction**: Single-channel IR ImageNet transfer must use `sum` reduction. `mean` is rejected and must never be passed to published runs.
3. **Class Names SSOT**: `classes.py:CLASS_NAMES` is the sole source of truth for the 10-class order.
4. **Shell Wrapper Mandate**: Never invoke bare `python` for Keras on the GPU server. Always run through `scripts/keras/*.sh` so `_keras_env.sh` sets `LD_LIBRARY_PATH` for `libcudnn.so.9`.
5. **NNAPI No-Fallback Policy**: Android runtime rejects a model slot on NNAPI setup/warmup failure; it must not fall back to CPU. Ensure all exported TFLite models conform strictly to NPU operators.
6. **User Concept Review Ownership**: `docs/keras-concept-review.md` tracks the user's comprehension. Never edit `이해 상태` on the user's behalf; update it only when explicitly requested by the user.
7. **Artifact Separation**: Store generated `.tflite`, `.keras`, and `.pth` only in gitignored `model/`. Never commit model weights or raw image datasets to Git.

## Build and Verification

```bash
# 1. Dataset split integrity & leakage check
python validate_fixed_splits.py

# 2. Fast unit tests (Company PC or GPU server)
pytest tests/dataset tests/metrics

# 3. Full test suite (GPU server with .venv-tf)
pytest

# 4. Keras smoke test
./scripts/keras/run_keras_model.sh

# 5. PyTorch & MCT setup check
./scripts/pytorch/run_pytorch_verify.sh

# 6. Universal TFLite evaluation on test split
python evaluate_tflite.py --model model/keras/best_crop_ir_fixed_npu_int8.tflite --split test
```
