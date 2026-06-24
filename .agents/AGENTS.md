# Project Rules and Guidelines

Behavioral and technical constraints specific to the `access-liveness-model` project.

## 1. Environment and Execution
- **WSL Virtual Environment:** All python scripts and packaging commands must be executed within the WSL environment (`/home/union/access-liveness-model`) using the `uv` package manager.
- **Execution Command Format:** Run scripts via `uv run python <script.py>` or `.venv/bin/python <script.py>`. Always execute via `wsl bash -c "cd /home/union/access-liveness-model && uv run ..."` when invoking from the parent agent shell.
- **Pre-execution Report:** Always explain to the user in Korean what command is being executed and why, prior to proposing or running the command.

## 2. Android Model Contract (Deployment Specifications)
Any TFLite model generated for deployment must strictly conform to the following specifications to ensure compatibility with `AntiSpoofingClassifier.java` without modifying the Android source code:
- **Inputs:** Exactly two (2) input tensors in NHWC format.
  - **Index 0 (RGB):** Shape `[1, 224, 224, 3]`, type `FLOAT32`.
  - **Index 1 (IR):** Shape `[1, 224, 224, 1]`, type `FLOAT32`.
- **Output:** Exactly one (1) output tensor.
  - **Shape:** `[1, 4]`, type `FLOAT32` (raw logits, `outputIsLogits: true` in `model_spec.json`).
- **Output Class Mapping (Fixed Indices):**
  - `[0]`: LIVE
  - `[1]`: SPOOF_MASK
  - `[2]`: DISPLAY
  - `[3]`: PHOTO

## 3. LiteRT-Torch & Layout Permutations
- **Channels-Last (NHWC) Conversion:** To achieve NHWC layout required by the Android NPU, always use `litert_torch.to_channel_last_io(model, args=[0, 1])` to wrap the PyTorch model before conversion.
- **Sample Inputs:** The tracing dummy inputs passed to `litert_torch.convert` must match the wrapped NHWC shapes (`[1, 224, 224, 3]` and `[1, 224, 224, 1]`) to prevent FX tracing dimension errors.

## 4. Output Directories
- **Gitignored Model Folder:** Export all generated model files (`*.tflite`, `*.onnx`, `*.pth`) to the project root `model/` directory (which is gitignored). Do not keep raw model weights in the project root directory.
