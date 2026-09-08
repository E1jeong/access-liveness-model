# Keras/TensorFlow Pipeline

This folder is isolated from the existing PyTorch pipeline.

It uses fixed train/validation/test directories:

```text
dataset/raw/{train,validation,test}/<class>/<class>_<subjectId>/<frame>/
  cropRGB.bmp
  cropIR.bmp
  RGB.bmp
  IR.bmp
```

`live` keeps its quality level:

```text
dataset/raw/{train,validation,test}/live/{high,medium}/live_<subjectId>/<frame>/
```

`train` is used for fitting and INT8 calibration, `validation` selects the
best checkpoint, and `test` is evaluated only after the configuration is
frozen. Run `python -m common.validate_fixed_splits` before training to reject missing
classes/files and subject leakage across splits.

The goal is to train a saved Keras model and convert it through the official
TensorFlow Lite converter path, which is a better fit for full INT8/NPU work
than the current PyTorch `.pth` to TFLite path.

## Files

- `data/`: input specification, pseudo-depth generation, and TensorFlow dataset construction.
- `models/`: input-contract builders, selectable backbones, and loss functions.
- `training/`: training entry point, artifact naming, and run metadata.
- `export/`: float/INT8/NPU-friendly TFLite conversion and graph validation.
- `contracts/`: Keras and TFLite input/output signature validation.
- `../common/validate_fixed_splits.py`: validates all three splits and blocks subject/frame leakage.

## Typical commands

Run on the sub-laptop GPU environment from the repository root. Use the
`scripts/keras/` wrappers because they set the TensorFlow CUDA library path
automatically:

```bash
./scripts/keras/run_keras_model.sh
.venv-tf/bin/python -m common.validate_fixed_splits
./scripts/keras/run_keras_train.sh --epochs 30
./scripts/keras/run_keras_convert.sh --float --int8 --npu-int8 --calibration-samples 500
.venv-tf/bin/python -m common.evaluate_tflite --split validation --models \
  model/keras/best_model_fixed_float.tflite \
  model/keras/best_model_fixed_int8.tflite
```

The end-to-end command is `./scripts/keras/run_fixed_split.sh`. It trains once, converts the
checkpoint, and evaluates `validation`; it never evaluates `test`
automatically. Final test evaluation must be requested explicitly:

```bash
.venv-tf/bin/python -m common.evaluate_tflite --split test --models \
  model/keras/best_model_fixed_float.tflite \
  model/keras/best_model_fixed_int8.tflite \
  model/keras/best_model_fixed_npu_int8.tflite
```

`--backbone` defaults to `mobilenetv2`; `efficientnet_lite0`
artifacts include their backbone name in every checkpoint,
TFLite, calibration manifest, learning-curve, and run-metadata filename.

The generated files go under `model/keras/` by default.

`--npu-int8` writes `model/keras/best_model_fixed_npu_int8.tflite`. It reuses the trained `.keras` weights and changes only the export graph:

- removes the RGB normalization Lambda from the TFLite graph,
- exports RGB input in MobileNet `[-1,1]` range,
- replaces `MEAN` global pooling with `AVERAGE_POOL_2D`,
- fixes batch size to 1 for Android deployment.

Android `model_spec.json` must match this export: RGB and IR both use `mean=[0.5]`, `std=[0.5]`. The standard float/int8 exports use RGB ImageNet mean/std instead.

Current target-board status is model-specific. The current twelve-class fixed-split `crop_ir` NPU-friendly INT8 model (`single_1_input` slot) has user-confirmed Android loading and stable operation. The prior combined candidate was observed at sustained 6.7–8 FPS; complete current-artifact delegate partition and latency measurements remain separate checks. Older paired six-class RGB fold3 and IR fold4 NPU-friendly INT8 models remain historical verified baselines. `dual` retraining is currently on hold per user decision. `Backend CPU` denotes an explicitly configured CPU slot; NNAPI setup/warmup failure rejects the slot without CPU fallback.

For the first MobileNetV2 ImageNet-weighted run, TensorFlow may need internet
access to download RGB backbone weights. If that is not available, run training
with:

```bash
./scripts/keras/run_keras_train.sh --rgb-weights none --no-gray-imagenet-init
```

Useful training switches:

- `--classifier-units 1024` is the default and mirrors the PyTorch classifier capacity more closely than a linear head.
- `--classifier-units 0` reverts to the old linear-head style for ablation.
- `--no-gray-imagenet-init` disables RGB-to-gray (IR) ImageNet weight transfer.
- `--label-smoothing 0.1` sets the label smoothing factor (default: 0.1) to prevent Softmax overconfidence and improve INT8 quantization stability.
- `--aux-supcon` adds the training-only supervised contrastive head. Defaults are
  `--supcon-loss-weight 0.1 --supcon-temperature 0.1 --projection-dim 128`;
  conversion strips this head and preserves the deployment logits contract.
