# Project Rules and Guidelines

Behavioral and technical constraints specific to the `access-liveness-model` project.

> Read `docs/project_guide.md` (fixed standards) and `docs/project_status.md` (current state, English) before working. `docs/overview_ko.md` is a plain-Korean summary for non-expert teammates.

## 0. Machine Topology (important)
Work spans two machines — do not assume one box:
- **Company machine** (this repo host): WSL Ubuntu, `.venv` (Python 3.11, **torch CPU**), created with `uv`. Holds code/docs/git. `dataset/raw` here is empty; data lives on the sub-laptop.
- **Sub-laptop** (GPU box): WSL2, GTX 1660 Ti, `.venv` (Python 3.12, **torch cu128**), plain `python -m venv`. SSH alias `mysub` (`won@e1jeong-home.duckdns.org:2222`). **All training, the dataset, and any quantization experiments run here.**
Transfer: edit on company machine → `rsync -avz <file> mysub:~/access-liveness-model/` → run on sub-laptop.

## 1. Environment and Execution
- **Use the project `.venv`.** On the company machine it was created with `uv` (run via `uv run python <script>` or `.venv/bin/python <script>`). On the sub-laptop use `source .venv/bin/activate` then `python <script>` (plain venv, no uv). Match whichever machine you are on.
- **Pre-execution Report:** Always explain to the user in Korean what command is being executed and why, prior to proposing or running the command.

## 2. Android Model Contract (Deployment Specifications)
Any TFLite model generated for deployment must strictly conform to the following specifications to ensure compatibility with `AntiSpoofingClassifier.java` without modifying the Android source code:
- **Inputs:** Exactly two (2) input tensors in NHWC format.
  - **Index 0 (RGB):** Shape `[1, 224, 224, 3]`, type `FLOAT32`.
  - **Index 1 (IR):** Shape `[1, 224, 224, 1]`, type `FLOAT32`.
- **Output:** Exactly one (1) output tensor.
  - **Shape:** `[1, 5]`, type `FLOAT32` (raw logits, `outputIsLogits: true` in `model_spec.json`).
- **Output Class Mapping (Fixed Indices):** Single source of truth is `classes.py` (`CLASS_NAMES`).
  - `[0]`: live
  - `[1]`: print
  - `[2]`: picture
  - `[3]`: mask
  - `[4]`: display
- **Normalization (must match `model_spec.json` and `dataset.py`):** RGB ImageNet mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]`; IR mean `[0.5]` / std `[0.5]`.

## 3. LiteRT-Torch & Layout Permutations
- **Channels-Last (NHWC) Conversion:** To achieve NHWC layout required by the Android NPU, always use `litert_torch.to_channel_last_io(model, args=[0, 1])` to wrap the PyTorch model before conversion.
- **Sample Inputs:** The tracing dummy inputs passed to `litert_torch.convert` must match the wrapped NHWC shapes (`[1, 224, 224, 3]` and `[1, 224, 224, 1]`) to prevent FX tracing dimension errors.

## 4. Output Directories and Deployment Handoff
- **Gitignored Model Folder:** Export all generated model files (`*.tflite`, `*.pth`) to the project root `model/` directory (which is gitignored). Do not keep raw model weights in the project root directory.
- **TFLite (float) is the current deployment format.** `convert_to_tflite.py` writes a float `model/anti_spoofing.tflite`. (ONNX is not used for deployment; it was used only in the INT8 investigation — see §5.)
- **Android handoff:** To deploy, manually copy `model/anti_spoofing.tflite` to the Android project's `app/src/main/assets/anti_spoofing.tflite`. The model in `assets/` is the committed deployment artifact; `model/` is gitignored.

## 5. Quantization / Deployment Status (read before any INT8 work)
- **Current deployment = float model on CPU.** Liveness ACER ≈ 0; CPU inference ~80–220 ms on the board.
- **INT8/NPU was attempted exhaustively and abandoned** (PTQ collapses on activations; PT2E QAT trains but cannot be serialized by litert_torch/eIQ; eIQ produced a broken tflite). Do NOT blindly retry the same paths — read the full chronology in `docs/project_status.md` §3 first.
- The Android app was reverted to **float-only** inference (INT8 I/O + NNAPI delegate removed; in git history if needed). The board NPU/NNAPI itself is confirmed ready for a future INT8 effort.
- If resuming INT8: prefer rebuilding the model in TensorFlow/Keras for eIQ's native QAT, or get NXP support. The QAT *training* worked; serialization is the blocker.
- **Data handoff:** Training images are collected on-device and delivered as files placed under `dataset/raw/<class>/<class>_<subjectId>/<frame>/` (`cropRGB.bmp`, `cropIR.bmp`, `RGB.bmp`, `IR.bmp`). This matches the Android collector output (`/sdcard/Pictures/raw/...`). There is no longer any webcam capture in this project.
