# Project Rules and Guidelines

Behavioral and technical constraints specific to the `access-liveness-model` project.

> This repo no longer has a `docs/` folder. Fixed standards, current state, and open questions live in the Obsidian vault (GitHub `E1jeong/obsidian-vault`, clone locally if not present) under `Project/Company/Access Liveness Model/`: `개요.md` (goals, fixed scope), `운영.md` (per-machine setup state, verification commands), `로드맵/개발 단계.md` (development gates), `테스트/평가지표와 결과.md` (current metrics), `이슈/확인 필요.md` (open items — read this first), `log.md` (append-only change history). The paired Android evaluation app has its own wiki at `Project/Company/Anti-Spoofing Viewer/` — check it whenever the model contract or NPU status changes, since the two repos must move together.

## 0. Machine Topology (important)
Work spans two machines — do not assume one box. This section is a fixed technical fact, not a status log; if it goes stale, fix it here directly (do not let the Obsidian wiki become the only source of truth for this repo's own machine names/paths).
- **Company machine** (this repo's edit host): WSL Ubuntu, `.venv` (Python 3.11, **torch CPU**), created with `uv`. Holds code/docs/git. `dataset/raw` here is empty; data lives on the sub-laptop.
- **Sub-laptop** (GPU box): native Ubuntu Server 24.04 (migrated from WSL2), GTX 1660 Ti, SSH alias `sub`. `.venv` (Python 3.12, `torch==2.11.0+cu128`) and `.venv-tf` (Python 3.11, `tensorflow[and-cuda]==2.21.0`), both created with **uv**. **All training, the dataset, and any quantization experiments run here.** Authoritative hardware/OS details: the Obsidian vault's `Server/서브노트북 (e1jeong)/` device wiki.
Transfer: edit on company machine → `rsync -avz <file> sub:~/access-liveness-model/` → run on sub-laptop.

## 1. Environment and Execution
- **Use the project `.venv`.** Both machines now create their venvs with `uv` — run via `uv run python <script>` or `.venv/bin/python <script>`.
- **Pre-execution Report:** Always explain to the user in Korean what command is being executed and why, prior to proposing or running the command.

## 2. Android Model Contract (Deployment Specifications)
Two model variants co-exist in this codebase, selected via `--model-type` (`keras_pipeline`) — see the Obsidian wiki's `기술/Android 배포 계약.md` for the full comparison table and known open questions (e.g. which variant `docs`-era performance numbers actually came from).

- **`dual` (2-input, legacy default)**: matches Android's `model_spec.json` (standard, CPU-only). Exactly two NHWC inputs:
  - **Index 0 (RGB):** Shape `[1, 224, 224, 3]`, type `FLOAT32` or `INT8`.
  - **Index 1 (IR):** Shape `[1, 224, 224, 1]`, type `FLOAT32` or `INT8`.
- **`multimodal` (5-input)**: matches Android's `model_spec_npu.json` (NPU delegate). Five NHWC inputs matched by substring against `model_spec_npu.json`'s `inputs.*` targets (not exact name equality) — see the Obsidian wiki for the exact tensor names Keras exports (`a_crop_rgb`, `b_crop_ir`, `c_rgb`, `d_ir`, `e_heatmap`).
- **Output (both variants):** Exactly one tensor, shape `[1, 5]`, type `FLOAT32` or `INT8` (raw logits, `outputIsLogits: true` in `model_spec*.json`).
- **Output Class Mapping (Fixed Indices):** Single source of truth is `classes.py` (`CLASS_NAMES`).
  - `[0]`: live
  - `[1]`: print
  - `[2]`: picture
  - `[3]`: mask
  - `[4]`: display
- **Normalization must match the exported model and the corresponding Android `model_spec*.json`:**
  - PyTorch/litert float and standard Keras export: RGB ImageNet mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]`; IR mean `[0.5]` / std `[0.5]`.
  - NPU-friendly Keras INT8 export (`*_npu_int8.tflite`): RGB and IR both use mean `[0.5]` / std `[0.5]`, so the model sees `[-1,1]` style inputs. This export removes the RGB Lambda preprocessing from the TFLite graph.

## 3. LiteRT-Torch & Layout Permutations
- **Channels-Last (NHWC) Conversion:** To achieve NHWC layout required by the Android NPU, always use `litert_torch.to_channel_last_io(model, args=[0, 1])` to wrap the PyTorch model before conversion. (This applies to the `pytorch_pipeline`/`dual` path only — it has no `multimodal` equivalent.)
- **Sample Inputs:** The tracing dummy inputs passed to `litert_torch.convert` must match the wrapped NHWC shapes (`[1, 224, 224, 3]` and `[1, 224, 224, 1]`) to prevent FX tracing dimension errors.

## 4. Output Directories and Deployment Handoff
- **Gitignored Model Folder:** Export all generated model files (`*.tflite`, `*.pth`) to the project root `model/` directory (which is gitignored). Do not keep raw model weights in the project root directory.
- **TFLite float and INT8 are both supported by the Android test app.** `convert_to_tflite.py` writes the PyTorch float path. `keras_pipeline/convert_h5_to_tflite.py --int8` writes standard Keras full INT8. `--npu-int8` writes the NPU-friendly full INT8 export.
- **Android handoff:** To deploy, manually copy `model/anti_spoofing.tflite` to the Android project's `app/src/main/assets/anti_spoofing.tflite`. The model in `assets/` is the committed deployment artifact; `model/` is gitignored.
- **Model artifacts are not synced by git.** Move `.keras` and `.tflite` files with `rsync`/`scp`, e.g. `rsync -avzR model/keras/best_model_fold4_npu_int8.tflite sub:~/access-liveness-model/`.

## 5. Quantization / Deployment Status (read before any INT8 work)
- **Current Android app attempts NNAPI first, then falls back to CPU/XNNPACK.** The on-screen backend label is authoritative: `Backend CPU` means NPU acceleration did not happen.
- **PyTorch/MobileNetV3 INT8 remains abandoned** (PTQ collapses on activations; PT2E QAT trains but cannot be serialized by litert_torch/eIQ; eIQ produced a broken tflite). Do NOT blindly retry the same paths — read the full chronology in the Obsidian wiki's `Project/Company/Access Liveness Model/기술/INT8 양자화와 NPU.md` first.
- **Keras/MobileNetV2 full INT8 conversion works and evaluates well locally.** See the Obsidian wiki's `테스트/평가지표와 결과.md` for current numbers — do not hardcode them here, they change.
- **NPU-friendly Keras INT8 export status may differ from what this file used to say.** The paired `Project/Company/Anti-Spoofing Viewer` wiki has reported the NPU-friendly export compiling on-device NNAPI successfully (as of a date later than this repo's own history) — verify against both wikis' `이슈/확인 필요.md` before reporting NPU status either way.
- Next NPU debugging should isolate unsupported ops from the remaining graph (`AVERAGE_POOL_2D`, `RESHAPE`, `CONCATENATION`, `FULLY_CONNECTED`, or quantized conv/depthwise constraints) instead of redoing training.
- **Data handoff:** Training images are collected on-device and delivered as files placed under `dataset/raw/<class>/<class>_<subjectId>/<frame>/` (`cropRGB.bmp`, `cropIR.bmp`, `RGB.bmp`, `IR.bmp`). This matches the Android collector output (`/sdcard/Pictures/raw/...`). There is no longer any webcam capture in this project. Note: the `multimodal` variant's 5th input (`heatmap`) reads an additional `face_heatmap.bmp` per frame if present, otherwise it is silently zero-filled — confirm this file actually exists in the current dataset before trusting any `multimodal` training result.
