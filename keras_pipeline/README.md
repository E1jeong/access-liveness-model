# Keras/TensorFlow Pipeline

This folder is isolated from the existing PyTorch pipeline.

It keeps the existing dataset layout unchanged:

```text
dataset/raw/<class>/<class>_<subjectId>/<frame>/
  cropRGB.bmp
  cropIR.bmp
  RGB.bmp
  IR.bmp
```

The goal is to train a saved Keras model and convert it through the official
TensorFlow Lite converter path, which is a better fit for full INT8/NPU work
than the current PyTorch `.pth` to TFLite path.

## Files

- `tf_dataset.py`: subject-wise K-fold dataset reader for the existing data. Spatial augmentation, resize, RGB ColorJitter, and normalization are aligned to the PyTorch pipeline.
- `tf_model.py`: dual-input MobileNetV2 Keras model. RGB uses ImageNet weights; IR can copy those weights by averaging the first RGB convolution to one channel.
- `train_tf.py`: trains and saves `.keras` checkpoints by best validation ACER.
- `convert_h5_to_tflite.py`: converts a saved Keras model to float or full INT8 TFLite.

## Typical commands

Run on the sub-laptop GPU environment. Use the root scripts because they set the
TensorFlow CUDA library path automatically:

```bash
./run_keras_model.sh
./run_keras_train.sh --epochs 30 --folds 5 --fold-idx 0
./run_keras_convert.sh --float --int8 --fold-idx 0
.venv/bin/python evaluate_tflite.py --models \
  model/keras/best_model_fold0_float.tflite \
  model/keras/best_model_fold0_int8.tflite
```

The generated files go under `model/keras/` by default.

For the first MobileNetV2 ImageNet-weighted run, TensorFlow may need internet
access to download RGB backbone weights. If that is not available, run training
with:

```bash
./run_keras_train.sh --rgb-weights none --no-ir-imagenet-init
```

Useful training switches:

- `--classifier-units 1024` is the default and mirrors the PyTorch classifier capacity more closely than a linear head.
- `--classifier-units 0` reverts to the old linear-head style for ablation.
- `--no-ir-imagenet-init` disables RGB-to-IR ImageNet weight transfer.
