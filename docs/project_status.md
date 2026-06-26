# Anti-Spoofing Project — Current Status (context for AI agents)

This file records changing facts and verification results. Fixed procedures/standards live in [project_guide.md](project_guide.md). Written in English for AI agents; a Korean non-expert summary is in [overview_ko.md](overview_ko.md).

- **Last updated**: 2026-06-26
- **Headline**: The float model works well (liveness ACER ≈ 0). INT8 quantization for the i.MX 8M Plus NPU was attempted exhaustively and **abandoned for now** — every available PyTorch/Google/NXP-eIQ path failed at a tool boundary. **Decision: ship the float model on CPU; treat INT8/NPU as a separate, properly-resourced future effort.**

## 0. Machine topology (important)

Work spans two machines. Do not assume everything is on one box.

- **Company machine** = this repo's host. WSL Ubuntu 24.04, project `.venv` (Python 3.11, **torch 2.12.1+cpu**). Used for code, docs, git, local data staging, and the Android project (Android project is a *separate* repo on the Windows side, see section 6). `dataset/raw/` currently contains the merged training dataset, but training here is CPU-only and slow. `model/` here holds no committed weights (gitignored).
- **Sub-laptop** = home GPU box. WSL2 Ubuntu, **GTX 1660 Ti 6GB**, driver CUDA 12.2. venv Python 3.12, **torch 2.11.0+cu128** (2.12.1 had no matching CUDA wheel). SSH alias `mysub` (`won@e1jeong-home.duckdns.org:2222`). **All real training and quantization experiments should run here** because GPU training is much faster. The merged dataset has also been rsynced here.
- **Target board** = i.MX 8M Plus running **Android** (accessed via `adb`). NPU = VeriSilicon (Vivante) VIP8000, INT8-only. NPU runtime confirmed present: `/dev/galcore`, `/vendor/lib64/{libGAL,libVSC,libnnrt,libovxlib,libOvx12VXCBinary-*}.so`, and `neuralnetworks_hal_vsi_npu_server: running`. So NPU acceleration is reachable via the Android **NNAPI delegate** once a working INT8 tflite exists.

Typical transfer: edit on company machine → `rsync -avz <file> mysub:~/access-liveness-model/` → run on sub-laptop. Pull artifacts back with `scp -P 2222 won@...:~/access-liveness-model/model/<f> ./model/`.

## 1. Status summary

### Verified (code/run evidence)
- `model.py` dual-input (RGB+IR) output is `[1,5]`.
- `classes.py` is the single source of classes: `0=live,1=print,2=picture,3=mask,4=display`.
- `dataset.py` splits subject-wise (`<class>_<id>` folder) K-fold; train/val non-overlap assert passes. Now has `num_workers`/`pin_memory`/`persistent_workers` (perf) and `get_data_loaders(..., num_workers=)`.
- `train.py` computes 5×5 confusion matrix, per-class recall, APCER/BPCER/ACER; saves best checkpoint by **lowest ACER**. Device is auto (`cuda` if available else `cpu`). DataLoader workers via `--num-workers`.
- **Float TFLite performance** (sub-laptop, merged dataset, fold-0 style validation, 1050 images): `val_acc=0.8905`, `APCER=0.0000`, `BPCER=0.0000`, `ACER=0.0000`. Per-class recall: `live=1.0000`, `print=0.4800`, `picture=0.9900`, `mask=1.0000`, `display=0.9550`. Liveness binary live-vs-spoof is excellent on this validation split, but `print` is weak as a 5-class subclass and is likely being confused with other spoof classes.
- Float tflite I/O (litert_torch, NHWC): inputs `[1,224,224,3]`+`[1,224,224,1]`, output `[1,5]`, all float32. Matches Android `model_spec.json` normalization (RGB ImageNet, IR 0.5/0.5).

### Not measured / not done
- Generalization to unseen people / lighting / distance. The merged dataset is larger, but the latest result is still validation/CV, not an independent field test.
- Independent test split (only K-fold CV).
- Dependency lock files.
- **INT8 / NPU latency** — see §3, abandoned.

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

## 5. Verification commands (train/evaluate on sub-laptop; local WSL is CPU-only)
```bash
source .venv/bin/activate
python model.py                         # output [1,5]
python train.py --folds 5               # K-fold train, prints APCER/BPCER/ACER
python convert_to_tflite.py             # float tflite -> model/anti_spoofing.tflite
python evaluate_tflite.py --models model/anti_spoofing.tflite   # eval a tflite on fold-0 val
```
`evaluate_tflite.py` auto-detects NCHW/NHWC input layout and handles float or int8 I/O; disables XNNPACK (reference kernels) with all CPU threads.

## 6. Android project
- Separate repo: `android-anti-spoofing-lab` (GitHub `E1jeong/android-anti-spoofing-lab`), on the Windows side at `C:\Users\Unionbiometrics\Desktop\company\2.source\ubio-anti-spoofing`.
- Inference: `app/src/main/java/com/virditech/ac7000/model/AntiSpoofingClassifier.java`, config `app/src/main/assets/model_spec.json` (rgbInputIndex/irInputIndex, channelOrder, mean/std, outputIsLogits, cropMarginRatio), TFLite 2.16.1.
- Reverted to float-only inference (see §4).

## 7. Known risks
- Reproducibility: no dependency lock; data/checkpoints/tflite are gitignored — repo alone cannot reproduce results.
- Small/possibly-homogeneous dataset (5 subjects) → liveness numbers may be optimistic; needs more subjects + varied capture conditions.
- INT8/NPU unverified (abandoned this round).

## 8. Change log
| Date | Change |
|---|---|
| 2026-06-25 | Cleaned webcam/ONNX-era remnants; rewrote docs to RGB+IR 5-class / device-capture / TFLite. |
| 2026-06-26 | GPU training on sub-laptop (5 subjects, float ACER about 0). Full INT8 investigation (PTQ collapse; QAT trains but cannot serialize; eIQ produces broken model) -> **INT8 abandoned, ship float-CPU**. Reverted Android to float-only. Deleted dead int8 scripts (train_qat.py, fix_onnx.py, export_onnx.py); convert_to_tflite.py reverted to float-only. Board NPU/NNAPI confirmed ready for a future INT8 effort. |
| 2026-06-26 | Documented dataset recrop/merge history: original `raw` 2500 sessions recropped with 10% margin; `raw2` 1349 sessions normalized, 10% recropped, and copy-merged into `raw` with `+5` folder offset. Verified current local `dataset/raw` totals: 3849 sessions / 15396 images. Latest float TFLite validation: `val_acc=0.8905`, `APCER/BPCER/ACER=0`; `print` recall remains weak at `0.4800`. |
