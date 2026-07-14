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
frozen. Run `validate_fixed_splits.py` before training to reject missing
classes/files and subject leakage across splits.

The goal is to train a saved Keras model and convert it through the official
TensorFlow Lite converter path, which is a better fit for full INT8/NPU work
than the current PyTorch `.pth` to TFLite path.

## Files

- `tf_dataset.py`: image loading and TensorFlow dataset construction. Spatial augmentation, resize, RGB ColorJitter, and normalization are aligned to the PyTorch pipeline.
- `tf_model.py`: dual-input MobileNetV2 Keras model. RGB uses ImageNet weights; IR can copy those weights by averaging the first RGB convolution to one channel.
- `tf_train.py`: trains and saves `.keras` checkpoints by best validation ACER.
- `convert_keras_to_tflite.py`: converts a saved Keras model to float, standard full INT8, or NPU-friendly full INT8 TFLite.
- `../validate_fixed_splits.py`: validates all three splits and blocks subject/frame leakage.

## Typical commands

Run on the sub-laptop GPU environment. Use the root scripts because they set the
TensorFlow CUDA library path automatically:

```bash
./run_keras_model.sh
.venv-tf/bin/python validate_fixed_splits.py
./run_keras_train.sh --epochs 30
./run_keras_convert.sh --float --int8 --npu-int8 --calibration-samples 500
.venv-tf/bin/python evaluate_tflite.py --split validation --models \
  model/keras/best_model_fixed_float.tflite \
  model/keras/best_model_fixed_int8.tflite
```

The end-to-end command is `./run_fixed_split.sh`. It trains once, converts the
checkpoint, and evaluates `validation`; it never evaluates `test`
automatically. Final test evaluation must be requested explicitly:

```bash
.venv-tf/bin/python evaluate_tflite.py --split test --models \
  model/keras/best_model_fixed_float.tflite \
  model/keras/best_model_fixed_int8.tflite \
  model/keras/best_model_fixed_npu_int8.tflite
```

The generated files go under `model/keras/` by default.

`--npu-int8` writes `model/keras/best_model_fixed_npu_int8.tflite`. It reuses the trained `.keras` weights and changes only the export graph:

- removes the RGB normalization Lambda from the TFLite graph,
- exports RGB input in MobileNet `[-1,1]` range,
- replaces `MEAN` global pooling with `AVERAGE_POOL_2D`,
- fixes batch size to 1 for Android deployment.

Android `model_spec.json` must match this export: RGB and IR both use `mean=[0.5]`, `std=[0.5]`. The standard float/int8 exports use RGB ImageNet mean/std instead.

Current target-board status is model-specific. The paired six-class RGB fold3
and IR fold4 NPU-friendly INT8 models have run with `Backend RGB NNAPI / IR
NNAPI`. New fixed-split exports, including `dual`, still require their own
on-device backend and latency verification. Treat `Backend CPU` as fallback,
not NPU acceleration.

For the first MobileNetV2 ImageNet-weighted run, TensorFlow may need internet
access to download RGB backbone weights. If that is not available, run training
with:

```bash
./run_keras_train.sh --rgb-weights none --no-gray-imagenet-init
```

Useful training switches:

- `--classifier-units 1024` is the default and mirrors the PyTorch classifier capacity more closely than a linear head.
- `--classifier-units 0` reverts to the old linear-head style for ablation.
- `--no-gray-imagenet-init` disables RGB-to-gray (IR/heatmap) ImageNet weight transfer.
- `--face-weights-path` loads a face-recognition-pretrained RGB backbone (`.weights.h5`/`.h5` via `Model.load_weights`, or `.npz` via the custom numpy loader in `tf_model.py`). These files (e.g. `model/mobilenetv2_mcp.pth`, `model/face_mobilenetv2_mcp.npz`, `model/pth_keys.txt`) are produced outside this repo and are gitignored under `model/` — see the Obsidian vault for provenance.
