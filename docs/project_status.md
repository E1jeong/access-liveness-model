# Anti-Spoofing Project — Current Status (context for AI agents)

This file records changing facts and verification results. Fixed procedures/standards live in [project_guide.md](project_guide.md). Written in English for AI agents; a Korean non-expert summary is in [overview_ko.md](overview_ko.md).

- **Last updated**: 2026-07-02
- **Headline**: The 5-input Keras multimodal npu_int8 model now **runs under NNAPI on the target board (user-confirmed 2026-07-02)**; warmup is ~10+ minutes (on-device NPU compilation), so NNAPI compilation caching is the next Android task. A 2026-07-02 pipeline review fixed an evaluation bug (npu_int8 models were fed ImageNet-normalized RGB instead of `[-1,1]`) and a live-only INT8 calibration bias — **all int8/npu_int8 metrics in §1 are stale until models are reconverted and re-evaluated on the sub-laptop.** The legacy PyTorch/MobileNetV3 code remains reference-only.

## 0. Machine topology (important)

Work spans two machines. Do not assume everything is on one box.

- **Sub-laptop** = **this repo's current host** (home GPU box). WSL2 Ubuntu, **GTX 1660 Ti 6GB**, NVIDIA driver **610.43.02** (upgraded 2026-06-28 from 535.98) / CUDA UMD 13.3 / CUDA toolkit 12.0 (`nvcc`). Root PyTorch scripts use `.venv` (Python 3.12, **torch 2.11.0+cu128**, `torch.cuda.is_available() == True`, confirmed). TensorFlow/Keras scripts use separate `.venv-tf` (Python 3.11, TensorFlow 2.21.0). **TF GPU fix confirmed 2026-06-28**: `tf.config.list_physical_devices('GPU')` returns the GTX 1660 Ti when `LD_LIBRARY_PATH` includes the nvidia package lib dirs inside `.venv-tf`. This is now handled automatically by `run_keras_*.sh` scripts — do not run `python keras_pipeline/train_tf.py` directly; use the shell scripts. Git is configured here; push to GitHub and pull on the company machine to sync code. `dataset/raw/` (training data) and `model/` (weights) are gitignored — sync these separately via rsync if needed on other machines. **All real training and quantization experiments run here.**
- **Company machine** = separate work PC. WSL Ubuntu 24.04, CPU-only torch. Used for Android project and code editing. Pull from GitHub (`git pull origin master`) to receive code updates made on the sub-laptop. The Android project is a *separate* repo on the Windows side (see §6).
- **Mac Studio** = additional model training machine (M1/M2/M3 Apple Silicon `arm64`, GPU MPS acceleration support). Added on 2026-07-02. Root working directory is `~/dev`. Can be accessed from the company WSL using `ssh mac` after setting up the SSH key authentication and SSH config alias (`mac`).
- **Target board** = i.MX 8M Plus running **Android** (accessed via `adb`). NPU = VeriSilicon (Vivante) VIP8000, INT8-only. NPU runtime confirmed present: `/dev/galcore`, `/vendor/lib64/{libGAL,libVSC,libnnrt,libovxlib,libOvx12VXCBinary-*}.so`, and `neuralnetworks_hal_vsi_npu_server: running`. So NPU acceleration is reachable via the Android **NNAPI delegate** once a working INT8 tflite exists.

Typical transfer: edit on company machine → `rsync -avz <file> mysub:~/access-liveness-model/` (for sub-laptop) or `rsync -avz <file> mac:~/dev/access-liveness-model/` (for Mac Studio).

## 0.1 Handoff for next session

Current stopping point on Thursday 2026-07-02 (company PC, code-only session):

- **On-device NNAPI success (user-confirmed 2026-07-02)**: the 5-input `best_model_fold1_npu_int8.tflite` runs on the i.MX 8M Plus board with the NNAPI delegate (no CPU fallback). Perceived warmup is 10+ minutes, attributed to on-device NPU model compilation; post-warmup latency/FPS are not measured yet.
- 2026-07-02 pipeline review fixes on branch `codex/keras-multimodal-deploy` (edited on the company PC, syntax-checked only — run verification pending on the sub-laptop):
  1. `evaluate_tflite.py` fed ImageNet-normalized RGB to NPU-friendly exports that expect MobileNet `[-1,1]` input. **All previously recorded npu_int8 metrics (§1 table) were measured under this mismatch and are stale.** The evaluator now picks the RGB range per model (`--rgb-range auto`: filename containing `npu` → `[-1,1]`).
  2. INT8 calibration used `train_items[:N]` on a class-ordered list, so calibration was ~100% `live` images. `convert_h5_to_tflite.py` now seed-shuffles items before sampling. **All int8/npu_int8 tflites should be reconverted.**
  3. `run_all_folds.sh` evaluated with `.venv/bin/python` (TF was removed from `.venv` on 2026-06-28 → would crash) and skipped `--npu-int8`. Now uses `.venv-tf` and converts/evaluates float+int8+npu_int8.
  4. `build_npu_export_model` hardcoded `classifier_units=1024`; now inferred from the trained checkpoint (also supports `--classifier-units 0`).
  5. Learning curves save per fold: `learning_curves_fold{N}.png` (was overwritten by each fold).
- Speed changes in the same review: `AcerCheckpoint` validation uses `model.predict` (compiled graph path; labels extracted once); `evaluate_tflite.py` evaluates all `--models` in one dataset pass with threaded sample prefetch (`--num-workers`, default 4); shared shell boilerplate extracted to `keras_env.sh`; experimental `--mixed-precision` train flag added (off by default; logits stay float32; TFLite-conversion compatibility unverified).
- Keras training still builds the 5-stream MobileNetV2 by default; inputs ordered `a_crop_rgb`, `b_crop_ir`, `c_rgb`, `d_ir`, `e_heatmap`. Batch size 8 / lr 1e-4 remains the safe baseline on the GTX 1660 Ti 6GB.
- Model artifacts are not synced by git. Use `rsync`/`scp` for `model/keras/*.keras` and `model/keras/*.tflite`; use git only for code/docs.

Next session order (all commands run on the sub-laptop):

1. Verify GPU is visible and the 5-input model still builds:
   ```bash
   ./run_keras_model.sh
   ```
2. Reconvert every fold with the fixed (shuffled) calibration sampling — checkpoints are unchanged, so no retraining is needed:
   ```bash
   ./run_keras_convert.sh --float --int8 --npu-int8 --fold-idx {N} --calibration-samples 500
   ```
3. Re-evaluate with the fixed evaluator (three models in a single pass, RGB range auto-selected):
   ```bash
   .venv-tf/bin/python evaluate_tflite.py --folds 5 --fold-idx {N} --models \
       model/keras/best_model_fold{N}_float.tflite \
       model/keras/best_model_fold{N}_int8.tflite \
       model/keras/best_model_fold{N}_npu_int8.tflite
   ```
   Replace the §1 table with the re-measured numbers and re-pick the deployment candidate.
4. `rsync`/`scp` the chosen npu_int8 model to the Android asset and confirm NNAPI still applies on the board.
5. Android warmup: enable NNAPI compilation caching (delegate `setCacheDir()` + `setModelToken()`) in `android-anti-spoofing-lab` so the 10+ minute first-run compilation is reused on subsequent runs.

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
- **Fold 4 Keras INT8 validation**: standard full INT8 TFLite `val_acc=0.9971`, `APCER=0.0000`, `BPCER=0.0120`, `ACER=0.0060`; NPU-friendly full INT8 TFLite `val_acc=0.9924`, `APCER=0.0000`, `BPCER=0.0320`, `ACER=0.0160`. NPU-friendly export is slightly worse on live recall (`0.9680` vs `0.9880`) but still useful for NPU execution experiments.
- NPU-friendly export structure: `best_model_fold4_npu_int8.tflite` has INT8 RGB/IR inputs `[1,224,224,3]` and `[1,224,224,1]` with quantization `(0.007843..., -1)`, INT8 output `[1,5]`, and no RGB preprocessing `MUL/ADD/SUB` Lambda ops or `MEAN` global pooling. Remaining non-conv ops include `AVERAGE_POOL_2D`, `RESHAPE`, `CONCATENATION`, and `FULLY_CONNECTED`.
- **5-input Keras multimodal code path (implemented 2026-07-01)**: `keras_pipeline/tf_dataset.py` returns `(cropRGB, cropIR, RGB, IR, heatmap)` tensors in NHWC format. `keras_pipeline/tf_model.py` builds five MobileNetV2 streams and concatenates 5 x 1280 features before the classifier. `train_tf.py`, `convert_h5_to_tflite.py`, and `evaluate_tflite.py` are wired to this 5-input contract by default.
- **5-input training startup & pipeline optimizations (2026-07-01)**: Optimized the Keras training dataset pipeline by replacing `from_generator` with `from_tensor_slices` combined with pre-shuffled path string caching, completely eliminating the startup lag from filling up the shuffle buffer. Applied validation caching (`.cache()`) and parallel preloading (via `ThreadPoolExecutor` with 8 workers) for calibration datasets to speed up model conversion.
- **5-Input Keras Multimodal 5-Fold Validation Results (2026-07-01) — ⚠ stale as of 2026-07-02**: measured before the evaluator RGB-range fix (npu_int8 rows were fed mismatched ImageNet-normalized RGB) and with live-only INT8 calibration. Re-convert and re-evaluate before quoting any int8/npu_int8 number. Original table:
  | Model / Fold | val_acc | APCER | BPCER | ACER |
  | :--- | :---: | :---: | :---: | :---: |
  | **Fold 0** | | | | |
  | - float | 0.9962 | 0.0025 | 0.0000 | 0.0013 |
  | - int8 | 0.9848 | 0.0013 | 0.0000 | 0.0006 |
  | - npu_int8 | 0.9886 | 0.0000 | 0.0120 | 0.0060 |
  | **Fold 1** | | | | |
  | - float | 0.9560 | 0.0000 | 0.0000 | 0.0000 |
  | - int8 | 0.9600 | 0.0000 | 0.0000 | 0.0000 |
  | - npu_int8 | 0.9630 | 0.0000 | 0.0000 | 0.0000 |
  | **Fold 2** | | | | |
  | - float | 0.9217 | 0.0000 | 0.0000 | 0.0000 |
  | - int8 | 0.9650 | 0.0000 | 0.0000 | 0.0000 |
  | - npu_int8 | 0.9600 | 0.0025 | 0.0000 | 0.0013 |
  | **Fold 3** | | | | |
  | - float | 0.9350 | 0.0000 | 0.0300 | 0.0150 |
  | - int8 | 0.9183 | 0.0000 | 0.0150 | 0.0075 |
  | - npu_int8 | 0.9267 | 0.0000 | 0.0000 | 0.0000 |
  | **Fold 4** | | | | |
  | - float | 0.9316 | 0.0000 | 0.0000 | 0.0000 |
  | - int8 | 0.9232 | 0.0000 | 0.0000 | 0.0000 |
  | - npu_int8 | 0.9299 | 0.0000 | 0.0000 | 0.0000 |
  * **Result analysis**: `npu_int8` models achieve an outstanding average liveness ACER of **0.15%** (APCER = 0.05%, BPCER = 0.24%). Fold 1 `npu_int8` is the prime candidate for deployment with perfect 0.00% liveness errors (ACER=0.00%, BPCER=0.00%, APCER=0.00%) and 96.30% 5-class validation accuracy.
- **Android Deployment (2026-07-01)**: Deployed `best_model_fold1_npu_int8.tflite` to the Android asset path `app/src/main/assets/anti_spoofing.tflite`. Updated `model_spec.json` config settings (`"rgbNormalization": "minus_one_to_one"`, `"delegate": "nnapi"`). Android app Java code already handles 5-input feeding and dynamically parses this normalization type, so no code change is required.
- **On-device NNAPI success (2026-07-02, user-reported)**: the deployed 5-input fold-1 npu_int8 model runs on the target board under the NNAPI delegate (no CPU fallback). Warmup is ~10+ minutes (on-device NPU compilation); post-warmup latency/FPS/memory are not measured yet. Next: NNAPI compilation caching.

### Field Test Results (2026-06-30)
- **Test Environment**: Building rooftop (outdoor, strong natural sunlight).
- **Key Findings**:
  - **Excellent Spoofing Class Discrimination**: The model distinguishes spoofing classes (print, picture, display, mask) very robustly.
  - **Liveness (live) Fluctuation beyond 1m**: Within 1m, the model correctly classifies `live`. However, when the subject is 1m or further away, the `live` classification tends to fluctuate/bounce.
- **Team Hypotheses & Proposed Mitigations**:
  - **IR Contrast Enhancement**: The team suspects that strong natural light outdoors enhances IR image quality/features, making spoofing characteristics highly visible to the model.
  - **Temporal Smoothing**: To address the liveness fluctuation beyond 1m, the team proposes implementing multi-frame voting/aggregation or modifying post-processing algorithms to filter out transient class-switching frames.

### Not measured / not done
- Generalization to unseen people / lighting / distance has had initial outdoor validation, but systematic testing across varied environments is still pending.
- Independent test split (only K-fold CV).
- Dependency lock files.
- **INT8 / NPU latency** — NNAPI execution confirmed on-device (2026-07-02), but post-warmup latency/FPS/memory are unmeasured. Warmup is ~10+ minutes; next step is NNAPI compilation caching (delegate `setCacheDir()`/`setModelToken()`).

## 2. Data status
- Structure `dataset/raw/<class>/<class>_<subjectId>/<frame>/` with `cropRGB.bmp,cropIR.bmp,RGB.bmp,IR.bmp,face_heatmap.bmp`. The 5-input Keras path uses all five files; the legacy PyTorch dual path still uses only `cropRGB.bmp,cropIR.bmp`.
- `face_heatmap.bmp` is expected to be a 224x224 single-channel grayscale BMP. It represents the detected face region prior. The Keras loader uses raw `0..1` scaling for heatmap and falls back to all-zero heatmap if a frame is missing the file.
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
- **Current Android test deployment is an INT8-capable app with NNAPI-first / CPU-XNNPACK fallback.** The checked Android asset may be the NPU-friendly Keras INT8 export; verify `app/src/main/assets/model_spec.json` before replacing the model.
- For the NPU-friendly Keras INT8 export, Android preprocessing must use RGB mean/std `[0.5]`/`[0.5]` and IR mean/std `[0.5]`/`[0.5]`. The standard Keras/PyTorch exports use RGB ImageNet normalization.
- **NNAPI acceleration is now working for the 5-input Keras npu_int8 export (user-confirmed on the board, 2026-07-02).** The historical `ANEURALNETWORKS_BAD_DATA ... while adding operation` failure applied to the 2-input era exports. If the UI ever shows `Backend CPU` again, treat it as a regression to CPU/XNNPACK fallback, not as NPU acceleration. Remaining on-device issue: 10+ minute warmup from NPU compilation → enable NNAPI compilation caching.

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

Root cause: `libcudnn.so.9` is installed inside `.venv-tf` pip packages (`site-packages/nvidia/cudnn/lib/`), not in system paths. TensorFlow cannot find it without `LD_LIBRARY_PATH`. The Keras shell scripts set this automatically from `.venv-tf`. PyTorch finds its CUDA libs internally and uses root `.venv` instead. The shared env setup (venv check, `LD_LIBRARY_PATH`, GPU print) lives in `keras_env.sh`, sourced by every `run_keras_*.sh`.

**Step 1 — verify GPU before training:**
```bash
./run_keras_model.sh          # prints GPU list and MobileNetV2 model summary
```
Expected: `GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

**Step 2 — train (one fold at a time):**
```bash
./run_keras_train.sh                                        # defaults: fold 0, 10 epochs
./run_keras_train.sh --epochs 10 --fold-idx 0              # current 5-input baseline
./run_keras_train.sh --epochs 20 --fold-idx 0              # longer run if best ACER is still improving
./run_keras_train.sh --folds 5 --fold-idx 1 --batch-size 16 --learning-rate 5e-5
```
`run_keras_train.sh` key args: `--epochs` `--fold-idx` `--folds` `--batch-size` `--learning-rate` `--seed` `--rgb-weights {imagenet,none}` `--dropout` `--classifier-units` `--no-gray-imagenet-init`. Deprecated alias: `--no-ir-imagenet-init`. Experimental: `--mixed-precision` (mixed_float16 training, logits kept float32; may allow larger batches on the GTX 1660 Ti, but TFLite conversion compatibility of the resulting checkpoint is unverified — validate before relying on it).
For the current 5-input model on GTX 1660 Ti 6GB, default batch size 8 / lr 1e-4 is the safe starting point. Batch size 16 caused startup/stability trouble in the observed run; only retry larger batches as a separate experiment and write to a separate `--output-dir`.

Default Keras model inputs:
- `a_crop_rgb`: crop RGB, ImageNet-normalized during training; NPU-friendly export expects MobileNet `[-1,1]`.
- `b_crop_ir`: crop IR, `mean=[0.5]`, `std=[0.5]`.
- `c_rgb`: full-frame RGB, ImageNet-normalized during training; NPU-friendly export expects MobileNet `[-1,1]`.
- `d_ir`: full-frame IR, `mean=[0.5]`, `std=[0.5]`.
- `e_heatmap`: face heatmap, raw `0..1`.

Outputs:
- Checkpoint: `model/keras/best_model_fold{N}.keras` (saved on best ACER each epoch)
- Learning curves: `model/keras/learning_curves_fold{N}.png`

**Step 3 — convert to TFLite:**
```bash
./run_keras_convert.sh --float --int8                                   # both modes, fold 0
./run_keras_convert.sh --float --int8 --fold-idx 1                     # fold 1 model
./run_keras_convert.sh --float                                          # float only
./run_keras_convert.sh --int8 --calibration-samples 300                # INT8, fewer samples
.venv/bin/python keras_pipeline/convert_h5_to_tflite.py --npu-int8 --fold-idx 4 --calibration-samples 500
```
`run_keras_convert.sh` key args: `--float` `--int8` `--npu-int8` `--fold-idx` `--model-path` `--output-dir` `--calibration-samples` (default 500). If shell scripts have line-ending issues on a given checkout, the direct `.venv/bin/python keras_pipeline/convert_h5_to_tflite.py ...` command is acceptable for conversion.

Outputs: `model/keras/best_model_fold{N}_float.tflite`, `model/keras/best_model_fold{N}_int8.tflite`, `model/keras/best_model_fold{N}_npu_int8.tflite`

For 5-input models, TFLite may list inputs in a different order than the Keras model. Do not bind by index only; bind by tensor name or by the exported `model_spec` contract.

**Step 4 — evaluate TFLite outputs:**
```bash
.venv-tf/bin/python evaluate_tflite.py --folds 5 --fold-idx 0 --models \
    model/keras/best_model_fold0_float.tflite \
    model/keras/best_model_fold0_int8.tflite \
    model/keras/best_model_fold0_npu_int8.tflite
```
All `--models` are evaluated in a **single pass** over the validation items: each sample is loaded once (threaded prefetch, `--num-workers` default 4) and fed to every interpreter, so comparing 3 models costs one dataset read. The RGB input range is chosen per model by `--rgb-range auto`: filenames containing `npu` get MobileNet `[-1,1]`, everything else gets ImageNet normalization (this is the 2026-07-02 fix for the npu_int8 preprocessing mismatch).
Note: `evaluate_tflite.py` falls back to `tensorflow.lite` if `ai_edge_litert` is not installed. Run it with `.venv-tf` (with `tqdm` installed) so the `keras_pipeline` imports of `tensorflow` resolve.

## 6. Android project
- Separate repo: `android-anti-spoofing-lab` (GitHub `E1jeong/android-anti-spoofing-lab`), on the Windows side at `C:\Users\Unionbiometrics\Desktop\company\2.source\ubio-anti-spoofing`.
- Inference: `app/src/main/java/com/virditech/ac7000/model/AntiSpoofingClassifier.java`, config `app/src/main/assets/model_spec.json` (rgbInputIndex/irInputIndex, channelOrder, mean/std, outputIsLogits, cropMarginRatio), TFLite 2.16.1.
- The app now feeds the 5-input contract (cropRGB/cropIR/RGB/IR/heatmap) and parses `rgbNormalization` from `model_spec.json` (2026-07-01), supporting FLOAT32/UINT8/INT8 inputs and FLOAT32/INT8 output `[1,5]`.
- NNAPI on the target board: **works with the 5-input npu_int8 export (user-confirmed 2026-07-02)**; warmup ~10+ min → next task is NNAPI compilation caching (`setCacheDir()`/`setModelToken()` on the delegate). Historical 2-input era failure for reference: `ANEURALNETWORKS_BAD_DATA at line 1131 while adding operation`.

## 7. Known risks
- Reproducibility: no dependency lock; data/checkpoints/tflite are gitignored — repo alone cannot reproduce results.
- Small/possibly-homogeneous dataset (5 subjects) → liveness numbers may be optimistic; needs more subjects + varied capture conditions.
- All recorded int8/npu_int8 accuracy numbers are stale (2026-07-02 evaluator RGB-range fix + calibration-bias fix) until models are reconverted and re-evaluated on the sub-laptop.
- On-device NNAPI works, but post-warmup latency/FPS/memory are unmeasured and the 10+ minute warmup is unresolved until NNAPI compilation caching lands.
- The 2026-07-02 code changes were made on the CPU-only company PC and are syntax-checked only; first sub-laptop run must confirm train/convert/evaluate still work end to end.

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
| 2026-06-29 | Android INT8/NNAPI path restored for testing: app accepts INT8 I/O, tries NNAPI first, and falls back to CPU/XNNPACK while showing `Backend CPU/NNAPI`. Standard Keras fold 4 INT8 validates well (`ACER=0.0060`). Added NPU-friendly export (`--npu-int8`) that removes RGB preprocessing Lambda ops and `MEAN` pooling; it validates at `ACER=0.0160` but still fails target-board NNAPI with `ANEURALNETWORKS_BAD_DATA ... while adding operation`, so NPU acceleration remains unsolved. |
| 2026-06-29 | `train_tf.py`: added `tf.config.experimental.set_memory_growth(gpu, True)` — TF now allocates VRAM on demand instead of pre-allocating all 6GB at startup. GPU utilization measured during batch_size=8 training: ~19% Epoch 1 (XLA compiling), ~64% Epoch 2+ (compiled). |
| 2026-06-29 | [pending] Batch size / learning rate scaling experiment not yet run. Current baseline: batch_size=8, lr=1e-4. Plan to test batch_size=32 with lr=4e-4 and batch_size=64 with lr=8e-4 (linear scaling rule). GPU utilization at batch_size=8 averages ~30%, so larger batches are expected to improve both training speed and GPU efficiency. |
| 2026-06-30 | Conducted outdoor field test (building rooftop). Spoofing detection was highly robust, potentially due to strong natural light enhancing IR features. Liveness detection was stable within 1m but fluctuated at >1m. Proposed multi-frame aggregation or heuristic smoothing as a mitigation. |
| 2026-07-01 | Added 5-input Keras multimodal pipeline on branch `codex/keras-multimodal-deploy`: dataset loads cropRGB/cropIR/RGB/IR/heatmap, model builds five MobileNetV2 streams, train/convert/evaluate scripts use the new input contract by default, and float/int8/npu-int8 conversion smoke tests passed with a random Keras model. Real 5-input training and Android 5-input integration remain pending. |
| 2026-07-01 | Started real 5-input fold-0 Keras training with the default batch size 8 / lr 1e-4 after batch size 16 proved impractical on the GTX 1660 Ti 6GB setup. |
| 2026-07-01 | Optimized Keras training pipeline by replacing single-threaded `from_generator` with `from_tensor_slices` and `tf.py_function(..., num_parallel_calls=tf.data.AUTOTUNE)`. Shuffling is now performed instantly on path strings before loading images, completely eliminating the `ShuffleDatasetV3: Filling up shuffle buffer` startup lag. Batch generation speed improved to ~0.08s per batch. Enabled RAM caching (`.cache()`) for validation dataset (`val_ds`), reducing end-of-epoch validation overhead from ~20s to ~1s starting from Epoch 2. |
| 2026-07-01 | Optimized Keras-to-TFLite conversion pipeline (`convert_h5_to_tflite.py`) by loading the heavy Keras model only once in `main()` instead of reloading it three times. Integrated multi-threaded parallel preloading (`ThreadPoolExecutor` with 8 workers) for calibration dataset samples to eliminate single-threaded disk I/O bottlenecks, significantly speeding up both standard and NPU-friendly INT8 conversions. |
| 2026-07-01 | Updated `evaluate_tflite.py` to support fallback to `tensorflow.lite.Interpreter` when `ai_edge_litert` is not present, allowing it to run within the Keras virtual environment (`.venv-tf`). Installed `tqdm` in `.venv-tf` on the sub-laptop, and updated all documentation references to run `evaluate_tflite.py` using `.venv-tf`. |
| 2026-07-02 | **On-device NNAPI confirmed** for the 5-input fold-1 npu_int8 model (user-reported; warmup 10+ min → plan NNAPI compilation cache). Pipeline review fixes (company PC, syntax-checked only): `evaluate_tflite.py` feeds MobileNet `[-1,1]` RGB to npu exports via `--rgb-range auto` (**previous npu_int8 metrics stale**) and evaluates all `--models` in one prefetched dataset pass; INT8 calibration items seed-shuffled before sampling (was ~100% live-class); `run_all_folds.sh` switched to `.venv-tf` and now includes `--npu-int8`; `build_npu_export_model` infers `classifier_units` from the checkpoint; per-fold learning-curve filenames; `AcerCheckpoint` validation via `model.predict`; shared `keras_env.sh`; experimental `--mixed-precision` train flag (logits kept float32). |



