# Anti-Spoofing Project — Current Status (context for AI agents)

This file records changing facts and verification results. Fixed procedures/standards live in [project_guide.md](project_guide.md). Written in English for AI agents; a Korean non-expert summary is in [overview_ko.md](overview_ko.md).

- **Last updated**: 2026-06-29
- **Headline**: The PyTorch float model works well (liveness ACER ≈ 0), and the current short-term deployment decision remains float-on-CPU. The previous PyTorch/MobileNetV3 INT8 path was abandoned after exhaustive toolchain failures. The TensorFlow/Keras MobileNetV2 path is now past the first real fold-0 run: full INT8 TFLite conversion and evaluation succeeded without collapse, but the 10-epoch result is not yet product-ready (`INT8 APCER=0.0250`, `BPCER=0.1080`, `ACER=0.0665`). The Keras training code has now been tightened to match the PyTorch recipe more closely; rerun fold 0 before judging Keras quality again.

## 0. Machine topology (important)

Work spans two machines. Do not assume everything is on one box.

- **Sub-laptop** = **this repo's current host** (home GPU box). WSL2 Ubuntu, **GTX 1660 Ti 6GB**, NVIDIA driver **610.43.02** (upgraded 2026-06-28 from 535.98) / CUDA UMD 13.3 / CUDA toolkit 12.0 (`nvcc`). PyTorch env `.venv` is Python 3.12 with **torch 2.11.0+cu128** (`torch.cuda.is_available() == True`, confirmed). TensorFlow/Keras uses separate `.venv-tf` (Python 3.11, TensorFlow 2.21.0). **TF GPU fix confirmed 2026-06-28**: `tf.config.list_physical_devices('GPU')` returns the GTX 1660 Ti when `LD_LIBRARY_PATH` includes the nvidia package lib dirs inside `.venv-tf`. This is now handled automatically by `run_keras_*.sh` scripts — do not run `python keras_pipeline/train_tf.py` directly; use the shell scripts. Git is configured here; push to GitHub and pull on the company machine to sync code. `dataset/raw/` (training data) and `model/` (weights) are gitignored — sync these separately via rsync if needed on other machines. **All real training and quantization experiments run here.**
- **Company machine** = separate work PC. WSL Ubuntu 24.04, CPU-only torch. Used for Android project and code editing. Pull from GitHub (`git pull origin master`) to receive code updates made on the sub-laptop. The Android project is a *separate* repo on the Windows side (see §6).
- **Target board** = i.MX 8M Plus running **Android** (accessed via `adb`). NPU = VeriSilicon (Vivante) VIP8000, INT8-only. NPU runtime confirmed present: `/dev/galcore`, `/vendor/lib64/{libGAL,libVSC,libnnrt,libovxlib,libOvx12VXCBinary-*}.so`, and `neuralnetworks_hal_vsi_npu_server: running`. So NPU acceleration is reachable via the Android **NNAPI delegate** once a working INT8 tflite exists.

Typical transfer: edit on company machine → `rsync -avz <file> mysub:~/access-liveness-model/` → run on sub-laptop. Pull artifacts back with `scp -P 2222 won@...:~/access-liveness-model/model/<f> ./model/`.

## 0.1 Handoff for next session

Current stopping point on Sunday 2026-06-29:

- NVIDIA driver upgraded to 610.43.02; TF GPU confirmed working via `LD_LIBRARY_PATH` fix.
- Full code refactor completed and pushed to GitHub (`master`). Repo is now git-managed; use `git pull` instead of rsync for code. Dataset/model weights are still gitignored and must be rsynced separately.
- Keras/TensorFlow pipeline (MobileNetV2, INT8 candidate) completed the first real fold-0 10-epoch run on the sub-laptop. Full INT8 TFLite conversion/evaluation succeeded without constant-class collapse, but metrics are not yet acceptable.
- Keras training parity fixes implemented: IR MobileNetV2 ImageNet weight transfer, 1024-unit hidden layer, augmentation order aligned with PyTorch, pre-shuffled train items.
- NPU-specific INT8 export path added to `convert_h5_to_tflite.py`: `build_npu_export_model()` (removes Lambda layer, uses explicit `AveragePooling2D`, `fixed_batch_size=1`), `convert_int8_npu()`, `--npu-int8` flag. Corresponding new params in `tf_model.py`: `rgb_input_mobilenet_range`, `average_pool_op`, `fixed_batch_size`. Not yet smoke-tested.
- `train_tf.py` now calls `tf.config.experimental.set_memory_growth(gpu, True)` — TF no longer pre-allocates all 6GB VRAM at startup.
- 30-epoch fold-0 Keras run not yet started.

Next session order (all commands run on the sub-laptop):

1. If starting from a fresh shell, verify GPU is still visible:
   ```bash
   ./run_keras_model.sh
   ```
2. Run or continue the longer Keras/MobileNetV2 fold-0 experiment:
   ```bash
   ./run_keras_train.sh --epochs 30 --fold-idx 0
   ```
3. Expected checkpoint: `model/keras/best_model_fold0.keras`.
4. Convert to TFLite (float + INT8) after training completes:
   ```bash
   ./run_keras_convert.sh --float --int8
   ```
5. Evaluate both float and INT8 outputs and compare:
   ```bash
   .venv/bin/python evaluate_tflite.py --models model/keras/best_model_fold0_float.tflite model/keras/best_model_fold0_int8.tflite
   ```

## 1. Status summary

### Verified (code/run evidence)
- `model.py` dual-input (RGB+IR) output is `[1,5]`.
- `classes.py` is the single source of classes: `0=live,1=print,2=picture,3=mask,4=display`.
- `dataset.py` splits subject-wise (`<class>_<id>` folder) K-fold; train/val non-overlap assert passes. Now has `num_workers`/`pin_memory`/`persistent_workers` (perf) and `get_data_loaders(..., num_workers=)`.
- `train.py` computes 5×5 confusion matrix, per-class recall, APCER/BPCER/ACER; saves best checkpoint by **lowest ACER**. Device is auto (`cuda` if available else `cpu`). DataLoader workers via `--num-workers`.
- Added isolated `keras_pipeline/` for TensorFlow/Keras saved-model -> TFLite experiments without modifying `dataset/raw` or the existing PyTorch pipeline. Initial `.h5` checkpoint saving failed on Keras/HDF5 duplicate dataset names, so the pipeline now saves native `.keras` checkpoints and the converter accepts `--model-path` (with `--h5-path` kept as an alias). Smoke-tested random-weight dual MobileNetV2 `.h5 -> float TFLite` and `.h5 -> full INT8 TFLite`; generated TFLite I/O order is RGB input index 0 `[1,224,224,3]`, IR input index 1 `[1,224,224,1]`, output `[1,5]`.
- **Float TFLite performance** (sub-laptop, merged dataset, fold-0 style validation, 1050 images): `val_acc=0.8905`, `APCER=0.0000`, `BPCER=0.0000`, `ACER=0.0000`. Per-class recall: `live=1.0000`, `print=0.4800`, `picture=0.9900`, `mask=1.0000`, `display=0.9550`. Liveness binary live-vs-spoof is excellent on this validation split, but `print` is weak as a 5-class subclass and is likely being confused with other spoof classes.
- Float tflite I/O (litert_torch, NHWC): inputs `[1,224,224,3]`+`[1,224,224,1]`, output `[1,5]`, all float32. Matches Android `model_spec.json` normalization (RGB ImageNet, IR 0.5/0.5).
- **First real Keras/MobileNetV2 fold-0 result** (sub-laptop, `./run_keras_train.sh --epochs 10 --fold-idx 0`, 1050-image validation): best Keras checkpoint reported `val_acc=0.7143`, `APCER=0.0612`, `BPCER=0.0160`, `ACER=0.0386`. Later epochs overfit/shifted toward rejecting live users (`epoch10 BPCER=0.2360`), so use the saved best checkpoint rather than the final epoch.
- **First Keras TFLite evaluation** from that checkpoint: float TFLite `val_acc=0.7295`, `APCER=0.0625`, `BPCER=0.0120`, `ACER=0.0372`; full INT8 TFLite `val_acc=0.7981`, `APCER=0.0250`, `BPCER=0.1080`, `ACER=0.0665`. INT8 did **not** collapse and has real int8 I/O (`RGB int8 [1,224,224,3]`, `IR int8 [1,224,224,1]`, output int8 `[1,5]`), but BPCER is too high and APCER is still above the 2% development target.
- **Keras parity fixes (implemented, not trained yet)**: `tf_model.py` now mirrors the PyTorch IR initialization pattern by copying ImageNet MobileNetV2 weights into the 1-channel IR backbone, and adds a default 1024-unit classifier hidden layer. `tf_dataset.py` now applies spatial augmentation before resize, ColorJitter after resize, and pre-shuffles train items before `tf.data` buffering to avoid class-blocked batches.

### Not measured / not done
- Longer Keras/MobileNetV2 training and/or all-fold validation. First fold-0 10-epoch run worked, but the result is preliminary.
- Generalization to unseen people / lighting / distance. The merged dataset is larger, but the latest result is still validation/CV, not an independent field test.
- Independent test split (only K-fold CV).
- Dependency lock files.
- **INT8 / NPU latency** — Keras INT8 TFLite now exists and runs in CPU/XNNPACK evaluation, but actual i.MX 8M Plus NNAPI/NPU execution and latency are not measured.

## 2. Data status
- Structure `dataset/raw/<class>/<class>_<subjectId>/<frame>/` with `cropRGB.bmp,cropIR.bmp,RGB.bmp,IR.bmp` (all four required by `dataset.py`). crop* are training inputs; RGB/IR are preserved originals.
- **Current merged real data: 3849 sessions / 15396 images** in `dataset/raw/` (verified on the company WSL after merge). Class folders:
  - `live`: `live_1` through `live_11` (11 subject folders)
  - `print`: `print_1` through `print_7` (7 subject folders)
  - `picture`: `picture_1` through `picture_7` (7 subject folders)
  - `mask`: `mask_1` through `mask_7` (7 subject folders)
  - `display`: `display_1` through `display_7` (7 subject folders)
- Dataset merge history reported by the previous agent:
  - Original `raw`: 5 classes x 5 subfolders x 100 sessions = 2500 sessions / 10000 images. Existing tight crops were expanded by a 10% margin using OpenCV template matching, overwriting the existing crop files. A 20% margin was tested and rejected.
  - New `raw2`: 1349 sessions. Folder names were irregular Korean/person labels, and IR files were 1-channel grayscale. It was normalized internally to numbered folders such as `live_1`, then 10% recropped while preserving 1-channel grayscale IR.
  - Final copy merge: `raw2` folders were copied into `raw` with a `+5` folder-number offset to avoid collisions, e.g. `raw2/live/live_1` -> `raw/live/live_6`. The reported merge covered 14 folders and did not modify the original `raw2` source.
- IR channel state is mixed by source and intentional: original `raw` IR images remain 3-channel RGB-mode files; `raw2`-derived IR images remain 1-channel grayscale files. `dataset.py` should continue reading IR as a single-channel model input.
- Reported data-cleanup artifacts: `walkthrough.md`, `batch_recrop.py`, `normalize_raw2.py`, `merge_datasets.py`, `mapping_log.json`, `verify_merged.py`. These were not found in the current repo root/docs bounded search on 2026-06-26, so treat them as external or uncommitted unless later located.
- K-fold requires subjects ≥ K. With the merged data, `--folds 5` is still valid. More subjects + varied capture conditions remain the main quality lever.

## 3. INT8 quantization investigation — full chronology (why it was abandoned)

Goal: INT8 tflite for the i.MX 8M Plus NPU (float on CPU is 80–220 ms; NPU INT8 would be ~5–20 ms). Every path failed:

1. **PTQ via `ai_edge_quantizer` (static a8)** → model **collapses**: outputs a constant class (always `display`) regardless of input. ACER 0.5.
2. **PTQ a16 (int16 activations)** → identical collapse → not an activation-bit-width issue.
3. **`--quant-mode w8only` (int8 weights, float activations, no calibration)** → **works** (ACER 0.0013 ≈ float). **Key diagnosis: int8 *weights* are fine; the collapse comes from *activation* PTQ** (MobileNetV3 hard-swish activations don't survive post-training activation quantization).
4. **PTQ a8 with 1000 calibration samples** → still collapses → not a calibration-quantity issue. PTQ is fundamentally unsuitable for this model.
5. **PT2E QAT (torchao XNNPACKQuantizer) + `litert_torch.convert`** → QAT **trains fine** (fake-quant val_acc 92–96% with per-channel) but the **litert converter fails to serialize the SE-block 1×1 convs** (`stablehlo.uniform_dequantize ... tensor<8x16x1x1xi8>`). litert_torch 0.9.1 is the latest version → no upgrade fix.
6. **Manual activation fake-quant QAT (forward hooks)** → **damaged the model**: BatchNorm running stats adapted to the fake-quant forward, so removing hooks broke it (live recall 0). Wrong approach for BN models.
7. **PT2E QAT (per-channel) → QDQ ONNX export** → `torch.onnx.export` **cannot emit `dequantize_per_channel`** → per-channel ONNX export unsupported in torch 2.11.
8. **PT2E QAT (per-tensor) → QDQ ONNX export** → **ONNX export succeeds** (4.18 MB) BUT per-tensor QAT accuracy is poor/unstable (val_acc bounced 38–84%, lr 1e-4 too high). per-tensor is forced because ONNX export only supports per-tensor QDQ.
9. **QDQ ONNX → NXP eIQ Toolkit (`eiq-converter-onnx2tflite`)**:
   - eIQ GUI quantizer (`eiq-converter-tflite` "Enable Quantization") only accepts Keras/TF SavedModel, not our ONNX-origin model → can't use eIQ's own PTQ. (Also eIQ PTQ would collapse like step 1 anyway — same TFLite PTQ.)
   - `onnx2tflite` of the QDQ ONNX first failed on `ReduceMean axes type INT32` (opset-18 axes-as-input form). Fixed via `fix_onnx.py` (converted ReduceMean axes input→attribute).
   - After the fix, conversion **"SUCCESS" but produced a structurally broken tflite**: `allocate_tensors` fails with `input_channel % filter_input_channel != 0 (1 != 0)` at a CONV_2D (caused by the `convert_reshape: flat size mismatch` warnings on the post-global-pool flatten). The output is NCHW + float I/O (not even int8 I/O), and does not run.

**Conclusion**: With this toolchain (PyTorch → ONNX/litert/eIQ) and this model (dual MobileNetV3-Small, hard-swish, SE blocks, dual input, custom per-channel normalization), getting a *working* INT8 tflite is not achievable by blind iteration. PTQ collapses; QAT trains but cannot be serialized cleanly.

### If INT8/NPU is resumed later — recommended directions (not yet attempted)
- **Rebuild the model in TensorFlow/Keras** and use eIQ's *native* QAT (the toolchain's supported happy path; eIQ quantization is TF-centric). This is the most likely to actually work end-to-end on i.MX.
- Or get **NXP engineering support** for the PyTorch→i.MX INT8 path.
- Or pick an architecture that PTQ-quantizes cleanly (avoid hard-swish / SE if NPU INT8 is a hard requirement).
- The QAT *training* code worked — the blocker is serialization, not the ML. Keep that in mind.

## 4. Current deployment decision
- **Ship the float model on CPU.** Android `AntiSpoofingClassifier.java` has been **reverted to float-only, CPU (numThreads 2)** — int8 I/O handling and NNAPI delegate were removed (they're in git history if needed). The float tflite (from `best_model_fold0.pth` via `convert_to_tflite.py`) goes in `app/src/main/assets/anti_spoofing.tflite`; `model_spec.json` is unchanged (float-compatible).
- Float CPU inference on the board is ~80–220 ms (functional, not fast). NPU acceleration is deferred to the future INT8 effort.

## 5. Verification commands (all run on the sub-laptop)

### PyTorch pipeline (`.venv`, Python 3.12)

```bash
.venv/bin/python model.py                              # smoke-test: prints output shape [1,5]
.venv/bin/python verify_setup.py                       # torch version, CUDA, litert_torch check
.venv/bin/python train.py --folds 5 --epochs 30        # K-fold train, all folds
.venv/bin/python train.py --folds 5 --max-folds 1      # single fold quick test
.venv/bin/python convert_to_tflite.py                  # float tflite -> model/anti_spoofing.tflite
.venv/bin/python evaluate_tflite.py --models model/anti_spoofing.tflite
```

`train.py` key args: `--epochs` `--batch-size` `--learning-rate` `--folds` `--max-folds` `--seed` `--num-workers`

### Keras/TensorFlow pipeline (`.venv-tf`, Python 3.11)

**Always use the shell scripts — never run `python keras_pipeline/…` directly.**

Root cause: `libcudnn.so.9` is installed only inside `.venv-tf` pip packages (`site-packages/nvidia/cudnn/lib/`), not in system paths. TensorFlow cannot find it without `LD_LIBRARY_PATH`. The shell scripts set this automatically. PyTorch finds its CUDA libs internally and does not have this requirement.

**Step 1 — verify GPU before training:**
```bash
./run_keras_model.sh          # prints GPU list and MobileNetV2 model summary
```
Expected: `GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

**Step 2 — train (one fold at a time):**
```bash
./run_keras_train.sh                                        # defaults: fold 0, 10 epochs
./run_keras_train.sh --epochs 30 --fold-idx 0              # 30 epochs, fold 0
./run_keras_train.sh --folds 5 --fold-idx 1 --batch-size 16 --learning-rate 5e-5
```
`run_keras_train.sh` key args: `--epochs` `--fold-idx` `--folds` `--batch-size` `--learning-rate` `--seed` `--rgb-weights {imagenet,none}` `--dropout` `--classifier-units` `--no-ir-imagenet-init`

Outputs:
- Checkpoint: `model/keras/best_model_fold{N}.keras` (saved on best ACER each epoch)
- Learning curves: `model/keras/learning_curves.png`

**Step 3 — convert to TFLite:**
```bash
./run_keras_convert.sh --float --int8                                   # both modes, fold 0
./run_keras_convert.sh --float --int8 --fold-idx 1                     # fold 1 model
./run_keras_convert.sh --float                                          # float only
./run_keras_convert.sh --int8 --calibration-samples 300                # INT8, fewer samples
./run_keras_convert.sh --npu-int8                                       # NPU/NNAPI-friendly INT8 (no Lambda, AveragePooling2D, batch=1)
```
`run_keras_convert.sh` key args: `--float` `--int8` `--npu-int8` `--fold-idx` `--model-path` `--output-dir` `--calibration-samples` (default 500)

`--npu-int8` vs `--int8` difference: `--npu-int8` uses `build_npu_export_model()` which removes the Lambda normalization layer (bakes normalization into the representative dataset instead) and replaces `pooling="avg"` with an explicit `AveragePooling2D` op — both changes improve NNAPI/eIQ compatibility. Output: `best_model_fold{N}_npu_int8.tflite`.

Outputs: `model/keras/best_model_fold{N}_float.tflite`, `model/keras/best_model_fold{N}_int8.tflite`

**Step 4 — evaluate TFLite outputs:**
```bash
.venv/bin/python evaluate_tflite.py \
    --models model/keras/best_model_fold0_float.tflite \
             model/keras/best_model_fold0_int8.tflite
```
Note: `evaluate_tflite.py` uses `.venv` (not `.venv-tf`) — it relies on `ai_edge_litert` which is in `.venv`.

## 6. Android project
- Separate repo: `android-anti-spoofing-lab` (GitHub `E1jeong/android-anti-spoofing-lab`), on the Windows side at `C:\Users\Unionbiometrics\Desktop\company\2.source\ubio-anti-spoofing`.
- Inference: `app/src/main/java/com/virditech/ac7000/model/AntiSpoofingClassifier.java`, config `app/src/main/assets/model_spec.json` (rgbInputIndex/irInputIndex, channelOrder, mean/std, outputIsLogits, cropMarginRatio), TFLite 2.16.1.
- Reverted to float-only inference (see §4).

## 7. Known risks
- Reproducibility: no dependency lock; data/checkpoints/tflite are gitignored — repo alone cannot reproduce results.
- Small/possibly-homogeneous dataset (5 subjects) → liveness numbers may be optimistic; needs more subjects + varied capture conditions.
- INT8/NPU unverified on the target board. Keras INT8 no longer collapses in local evaluation, but board NNAPI/NPU execution and latency remain unmeasured.

## 8. Change log
| Date | Change |
|---|---|
| 2026-06-25 | Cleaned webcam/ONNX-era remnants; rewrote docs to RGB+IR 5-class / device-capture / TFLite. |
| 2026-06-26 | GPU training on sub-laptop (5 subjects, float ACER about 0). Full INT8 investigation (PTQ collapse; QAT trains but cannot serialize; eIQ produces broken model) -> **INT8 abandoned, ship float-CPU**. Reverted Android to float-only. Deleted dead int8 scripts (train_qat.py, fix_onnx.py, export_onnx.py); convert_to_tflite.py reverted to float-only. Board NPU/NNAPI confirmed ready for a future INT8 effort. |
| 2026-06-26 | Documented dataset recrop/merge history: original `raw` 2500 sessions recropped with 10% margin; `raw2` 1349 sessions normalized, 10% recropped, and copy-merged into `raw` with `+5` folder offset. Verified current local `dataset/raw` totals: 3849 sessions / 15396 images. Latest float TFLite validation: `val_acc=0.8905`, `APCER/BPCER/ACER=0`; `print` recall remains weak at `0.4800`. |
| 2026-06-26 | Added isolated TensorFlow/Keras path under `keras_pipeline/`: existing dataset reader, dual-input MobileNetV2 `.h5` training, and `.h5 -> float/full-INT8 TFLite` conversion. Local smoke tests passed for model construction, dataset split, `.h5 -> float TFLite`, and `.h5 -> full INT8 TFLite` with random weights and small calibration sample. Accuracy/NPU delegate execution are not yet measured. |
| 2026-06-26 | Keras path first real sub-laptop run exposed environment and script issues: WSL sees GTX 1660 Ti via `/usr/lib/wsl/lib/nvidia-smi`, PyTorch still uses GPU in `.venv`, but TensorFlow 2.21.0 reports no GPU in both the existing `.venv` and new `.venv-tf`; likely TensorFlow CUDA package vs NVIDIA driver/runtime mismatch. CPU training also hit finite `tf.data.Dataset` exhaustion and HDF5 `.h5` save-name collision; `train_tf.py` now repeats train dataset with explicit steps and saves `.keras`, while converter uses `--model-path`. |
| 2026-06-28 | **NVIDIA driver upgraded** on sub-laptop from 535.98 to 610.43.02 (CUDA UMD 13.3). **TF GPU fix**: `tf.config.list_physical_devices('GPU')` now returns GTX 1660 Ti when `LD_LIBRARY_PATH` includes `.venv-tf` nvidia package lib dirs; root cause was TF not searching pip-installed CUDA paths automatically (unlike PyTorch). |
| 2026-06-28 | Code refactor across all Python files: (1) `utils.py` created — K-fold helpers, `gather_frame_items`, `calculate_validation_metrics` unified for both pipelines; (2) `model.py` — IR backbone pretrained weight transfer bug fixed (was random-init, now averages 3-ch weights to 1-ch), `Dropout(inplace=True)` removed; (3) `dataset.py` / `tf_dataset.py` — joint RGB+IR spatial augmentation (flip, rotation) and RGB ColorJitter added; (4) `train.py` — `CosineAnnealingLR` scheduler added; (5) `train_tf.py` — duplicate validation forward pass per epoch removed; (6) `convert_*.py` — `os.makedirs("")` crash fixed. |
| 2026-06-28 | `run_keras_model.sh`, `run_keras_train.sh`, `run_keras_convert.sh` added — wrap `LD_LIBRARY_PATH` setup so TF GPU works without manual env export. |
| 2026-06-28 | Git repository initialized on sub-laptop and pushed to GitHub (`E1jeong/access-liveness-model`, `master`). Previous commit history was not preserved (force push from unrelated history). Windows Git Credential Manager connected to WSL for authentication. |
| 2026-06-28 | Keras pipeline synced with PyTorch pipeline: `tf_dataset.py` — rotation ±10° augmentation added (was missing), ColorJitter aligned to PyTorch params (brightness/contrast [0.7,1.3], saturation [0.8,1.2] added); `train_tf.py` — CosineDecay LR added (alpha=0.01, matches PyTorch CosineAnnealingLR), APCER self-check added, learning curve save added (`model/keras/learning_curves.png`). `matplotlib` added to `.venv-tf`. |
| 2026-06-28 | `.venv` cleaned: `tensorflow` and `keras` removed (were manually installed during early TF-in-PyTorch-venv experiment; not required by any current dependency). `.venv` PyTorch pipeline verified intact after removal. §5 expanded with full script arguments, GPU root-cause explanation, and output file locations. |
| 2026-06-29 | First real Keras/MobileNetV2 fold-0 10-epoch run completed on the sub-laptop. Best checkpoint: `val_acc=0.7143`, `APCER=0.0612`, `BPCER=0.0160`, `ACER=0.0386`; final epochs overfit/shifted toward higher BPCER, so best checkpoint matters. Converted both float and full INT8 TFLite. Float TFLite: `val_acc=0.7295`, `APCER=0.0625`, `BPCER=0.0120`, `ACER=0.0372`. INT8 TFLite: `val_acc=0.7981`, `APCER=0.0250`, `BPCER=0.1080`, `ACER=0.0665`. INT8 conversion/evaluation did not collapse, but metrics are not yet product-ready and target-board NPU latency is still unmeasured. |
| 2026-06-29 | Keras training recipe tightened after comparing against PyTorch: IR MobileNetV2 now receives ImageNet weight transfer from the RGB backbone, the Keras classifier defaults to a 1024-unit hidden layer, augmentation order is aligned with PyTorch, and train item order is pre-shuffled before `tf.data` buffering. Smoke checks passed for Python compilation, Keras model construction (`output_shape=(None,5)`, 7,142,981 params), IR weight-copy count (104 layers), and a mixed-class shuffled first batch. Full retraining still required. |
| 2026-06-29 | NPU INT8 export path added to `convert_h5_to_tflite.py`: `build_npu_export_model()` rebuilds the model without the Lambda normalization layer and with an explicit `AveragePooling2D` op + `fixed_batch_size=1` for NNAPI/eIQ compatibility; `convert_int8_npu()` runs PTQ calibration on this export model; `--npu-int8` CLI flag added to `run_keras_convert.sh`. New params in `tf_model.py`: `rgb_input_mobilenet_range`, `average_pool_op`, `fixed_batch_size`. Not yet smoke-tested. `overview_ko.md` updated with MobileNetV2 vs V3 quantization explanation and NPU speed comparison. |
| 2026-06-29 | `train_tf.py`: added `tf.config.experimental.set_memory_growth(gpu, True)` — TF now allocates VRAM on demand instead of pre-allocating all 6GB at startup. Confirmed via `nvidia-smi` monitoring during a 3-epoch test run (batch_size=8): VRAM peaked at ~4.8GB during that run; with memory growth enabled, expected peak will reflect actual model+batch usage only. |
