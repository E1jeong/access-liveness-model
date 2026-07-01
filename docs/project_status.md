# Anti-Spoofing Project — Current Status (context for AI agents)

This file records changing facts and verification results. Fixed procedures/standards live in [project_guide.md](project_guide.md). Written in English for AI agents; a Korean non-expert summary is in [overview_ko.md](overview_ko.md).

- **Last updated**: 2026-07-01
- **Headline**: Keras/MobileNetV2 has been moved to a 5-input multimodal pipeline by default: `cropRGB`, `cropIR`, full-frame `RGB`, full-frame `IR`, and `face_heatmap`. The legacy PyTorch/MobileNetV3 code remains reference-only for this work. Float, full INT8, and NPU-friendly INT8 conversion paths have been updated to the 5-input contract and smoke-tested with a random Keras model. Historical fold 4 2-input INT8 metrics are still the latest real trained-model metrics; retraining the new 5-input model is required before comparing accuracy.

## 0. Machine topology (important)

Work spans two machines. Do not assume everything is on one box.

- **Sub-laptop** = **this repo's current host** (home GPU box). WSL2 Ubuntu, **GTX 1660 Ti 6GB**, NVIDIA driver **610.43.02** (upgraded 2026-06-28 from 535.98) / CUDA UMD 13.3 / CUDA toolkit 12.0 (`nvcc`). Root PyTorch scripts use `.venv` (Python 3.12, **torch 2.11.0+cu128**, `torch.cuda.is_available() == True`, confirmed). TensorFlow/Keras scripts use separate `.venv-tf` (Python 3.11, TensorFlow 2.21.0). **TF GPU fix confirmed 2026-06-28**: `tf.config.list_physical_devices('GPU')` returns the GTX 1660 Ti when `LD_LIBRARY_PATH` includes the nvidia package lib dirs inside `.venv-tf`. This is now handled automatically by `run_keras_*.sh` scripts — do not run `python keras_pipeline/train_tf.py` directly; use the shell scripts. Git is configured here; push to GitHub and pull on the company machine to sync code. `dataset/raw/` (training data) and `model/` (weights) are gitignored — sync these separately via rsync if needed on other machines. **All real training and quantization experiments run here.**
- **Company machine** = separate work PC. WSL Ubuntu 24.04, CPU-only torch. Used for Android project and code editing. Pull from GitHub (`git pull origin master`) to receive code updates made on the sub-laptop. The Android project is a *separate* repo on the Windows side (see §6).
- **Target board** = i.MX 8M Plus running **Android** (accessed via `adb`). NPU = VeriSilicon (Vivante) VIP8000, INT8-only. NPU runtime confirmed present: `/dev/galcore`, `/vendor/lib64/{libGAL,libVSC,libnnrt,libovxlib,libOvx12VXCBinary-*}.so`, and `neuralnetworks_hal_vsi_npu_server: running`. So NPU acceleration is reachable via the Android **NNAPI delegate** once a working INT8 tflite exists.

Typical transfer: edit on company machine → `rsync -avz <file> mysub:~/access-liveness-model/` → run on sub-laptop. Pull artifacts back with `scp -P 2222 won@...:~/access-liveness-model/model/<f> ./model/`.

## 0.1 Handoff for next session

Current stopping point on Wednesday 2026-07-01:

- New branch: `codex/keras-multimodal-deploy`.
- PyTorch root files are intentionally untouched; Keras is now the active implementation target.
- Keras training now builds a 5-stream MobileNetV2 model by default. Inputs are ordered at the Keras model boundary as `a_crop_rgb`, `b_crop_ir`, `c_rgb`, `d_ir`, `e_heatmap`.
- `tf_dataset.py` now loads full-frame RGB/IR and `face_heatmap.bmp` alongside the crop inputs. Missing or unreadable heatmaps fall back to a black 224x224 heatmap, matching the earlier PyTorch multimodal loader policy.
- TFLite may reorder model inputs in the exported flatbuffer. INT8 representative datasets now yield name-keyed dictionaries, and `evaluate_tflite.py` maps inputs by tensor name instead of channel count.
- Random-weight smoke checks passed for model construction, dataset batch creation, float conversion, full INT8 conversion, NPU-friendly INT8 conversion, and small-sample TFLite evaluation. A real 5-input training run has not been completed yet.
- First real 5-input fold-0 training was started with `./run_keras_train.sh --epochs 10 --fold-idx 0` on the sub-laptop. Batch size 8 / lr 1e-4 is the current safe baseline; batch size 16 / lr 2e-4 stalled or was impractical on the GTX 1660 Ti 6GB environment.
- **Optimized dataset pipeline (2026-07-01)**: Solved dataset pipeline loading and shuffling bottlenecks. Replaced the single-threaded `tf.data.Dataset.from_generator` with `from_tensor_slices` and `tf.py_function(..., num_parallel_calls=tf.data.AUTOTUNE)`. Shuffling is now performed instantly on path strings before loading images, eliminating `ShuffleDatasetV3: Filling up shuffle buffer` delays. Batch generation time is reduced to ~0.08s.
- Model artifacts are not synced by git. Use `rsync`/`scp` for `model/keras/*.keras` and `model/keras/*.tflite`; use git only for code/docs.

Next session order (all commands run on the sub-laptop):

1. If starting from a fresh shell, verify GPU is still visible and the 5-input model builds:
   ```bash
   ./run_keras_model.sh
   ```
2. Train the new 5-input model before making any accuracy claim:
   ```bash
   ./run_keras_train.sh --epochs 10 --fold-idx 0
   ```
   Use the default batch size 8 / learning rate 1e-4 first. Do not assume batch size 16 will fit or start quickly on the GTX 1660 Ti 6GB setup.
3. Convert the trained 5-input checkpoint:
   ```bash
   ./run_keras_convert.sh --float --int8 --npu-int8 --fold-idx 0 --calibration-samples 500
   ```
4. Evaluate the converted outputs:
   ```bash
   .venv/bin/python evaluate_tflite.py --models \
       model/keras/best_model_fold0_float.tflite \
       model/keras/best_model_fold0_int8.tflite \
       model/keras/best_model_fold0_npu_int8.tflite
   ```
5. Android deployment must be updated to feed five tensors before the new model can run on-device. The old Android 2-input asset contract is not compatible with this 5-input branch.

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
- **5-input smoke tests (2026-07-01)**: Python compile passed for Keras dataset/model/train/convert and `evaluate_tflite.py`; `tf_model.py --rgb-weights none --no-gray-imagenet-init` built the 5-input model and produced output shape `(1,5)`; a real dataset batch produced shapes `[2,224,224,3]`, `[2,224,224,1]`, `[2,224,224,3]`, `[2,224,224,1]`, `[2,224,224,1]`; random `.keras` model conversion passed for float, full INT8, and NPU-friendly INT8; `evaluate_tflite.py --max-samples 2` ran on random float/int8 TFLite outputs. These are structural smoke tests, not accuracy measurements.
- **5-input training startup notes (2026-07-01)**: `./run_keras_train.sh --epochs 10 --fold-idx 0` reached `Epoch 1/10` and initialized CUDA/XLA/cuDNN on the GTX 1660 Ti. The training pipeline was optimized on 2026-07-01 to eliminate `ShuffleDatasetV3: Filling up shuffle buffer` lag. Batch size 16 is not the current baseline for this 5-stream model; use batch size 8 unless a later run proves a larger batch is stable.

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
- **INT8 / NPU latency** — Keras INT8 TFLite exists and runs in CPU/XNNPACK evaluation, but actual i.MX 8M Plus NNAPI/NPU execution has not succeeded. Current Android result is `Backend CPU` fallback with `ANEURALNETWORKS_BAD_DATA`.

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
- **Do not claim NPU acceleration yet.** Current target-board logs show `ANEURALNETWORKS_BAD_DATA ... while adding operation`, and the UI remains `Backend CPU`. CPU fallback is expected until the unsupported operation/quantization constraint is isolated.

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

Root cause: `libcudnn.so.9` is installed inside `.venv-tf` pip packages (`site-packages/nvidia/cudnn/lib/`), not in system paths. TensorFlow cannot find it without `LD_LIBRARY_PATH`. The Keras shell scripts set this automatically from `.venv-tf`. PyTorch finds its CUDA libs internally and uses root `.venv` instead.

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
`run_keras_train.sh` key args: `--epochs` `--fold-idx` `--folds` `--batch-size` `--learning-rate` `--seed` `--rgb-weights {imagenet,none}` `--dropout` `--classifier-units` `--no-gray-imagenet-init`. Deprecated alias: `--no-ir-imagenet-init`.
For the current 5-input model on GTX 1660 Ti 6GB, default batch size 8 / lr 1e-4 is the safe starting point. Batch size 16 caused startup/stability trouble in the observed run; only retry larger batches as a separate experiment and write to a separate `--output-dir`.

Default Keras model inputs:
- `a_crop_rgb`: crop RGB, ImageNet-normalized during training; NPU-friendly export expects MobileNet `[-1,1]`.
- `b_crop_ir`: crop IR, `mean=[0.5]`, `std=[0.5]`.
- `c_rgb`: full-frame RGB, ImageNet-normalized during training; NPU-friendly export expects MobileNet `[-1,1]`.
- `d_ir`: full-frame IR, `mean=[0.5]`, `std=[0.5]`.
- `e_heatmap`: face heatmap, raw `0..1`.

Outputs:
- Checkpoint: `model/keras/best_model_fold{N}.keras` (saved on best ACER each epoch)
- Learning curves: `model/keras/learning_curves.png`

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
.venv/bin/python evaluate_tflite.py \
    --models model/keras/best_model_fold0_float.tflite \
             model/keras/best_model_fold0_int8.tflite
```
For the current fold 4 INT8 comparison:
```bash
.venv/bin/python evaluate_tflite.py \
    --models model/keras/best_model_fold4_int8.tflite \
             model/keras/best_model_fold4_npu_int8.tflite
```
Note: `evaluate_tflite.py` was verified with `.venv` because `ai_edge_litert` is installed there in the current checkout.

## 6. Android project
- Separate repo: `android-anti-spoofing-lab` (GitHub `E1jeong/android-anti-spoofing-lab`), on the Windows side at `C:\Users\Unionbiometrics\Desktop\company\2.source\ubio-anti-spoofing`.
- Inference: `app/src/main/java/com/virditech/ac7000/model/AntiSpoofingClassifier.java`, config `app/src/main/assets/model_spec.json` (rgbInputIndex/irInputIndex, channelOrder, mean/std, outputIsLogits, cropMarginRatio), TFLite 2.16.1.
- Current app supports FLOAT32/UINT8/INT8 inputs and FLOAT32/INT8 output `[1,5]` for the older 2-input contract. The new 5-input Keras branch requires Android-side changes before deployment: add full RGB, full IR, and heatmap input tensors and bind them by tensor name/order from the exported model.
- Current target-board NNAPI failure: `java.lang.IllegalArgumentException: Internal error: Failed to apply delegate: NN API returned error ANEURALNETWORKS_BAD_DATA at line 1131 while adding operation.`

## 7. Known risks
- Reproducibility: no dependency lock; data/checkpoints/tflite are gitignored — repo alone cannot reproduce results.
- Small/possibly-homogeneous dataset (5 subjects) → liveness numbers may be optimistic; needs more subjects + varied capture conditions.
- INT8/NPU currently fails on the target board and falls back to CPU. The next risk-reduction step is isolating whether `AVERAGE_POOL_2D`, `RESHAPE`, `CONCATENATION`, `FULLY_CONNECTED`, or a quantized conv/depthwise parameter constraint triggers `ANEURALNETWORKS_BAD_DATA`.

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

