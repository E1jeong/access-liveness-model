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

The goal is to train a Keras `.h5` model and convert it through the official
TensorFlow Lite converter path, which is a better fit for full INT8/NPU work
than the current PyTorch `.pth` to TFLite path.

## Files

- `tf_dataset.py`: subject-wise K-fold dataset reader for the existing data.
- `tf_model.py`: dual-input MobileNetV2 Keras model.
- `train_tf.py`: trains and saves `.h5` checkpoints.
- `convert_h5_to_tflite.py`: converts `.h5` to float or full INT8 TFLite.

## Typical commands

Run on the sub-laptop GPU environment:

```bash
source .venv/bin/activate
python keras_pipeline/train_tf.py --epochs 10 --folds 5 --fold-idx 0
python keras_pipeline/convert_h5_to_tflite.py --h5-path model/keras/best_model_fold0.h5 --float
python keras_pipeline/convert_h5_to_tflite.py --h5-path model/keras/best_model_fold0.h5 --int8
```

The generated files go under `model/keras/` by default.

For the first MobileNetV2 ImageNet-weighted run, TensorFlow may need internet
access to download RGB backbone weights. If that is not available, run training
with:

```bash
python keras_pipeline/train_tf.py --rgb-weights none
```
