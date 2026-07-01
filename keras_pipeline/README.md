# Keras/TensorFlow Pipeline

This folder is isolated from the existing PyTorch pipeline.

It keeps the existing dataset layout unchanged:

```text
dataset/raw/<class>/<class>_<subjectId>/<frame>/
  cropRGB.bmp
  cropIR.bmp
  RGB.bmp
  IR.bmp
  face_heatmap.bmp
```

The goal is to train a saved Keras model and convert it through the official
TensorFlow Lite converter path, which is a better fit for full INT8/NPU work
than the current PyTorch `.pth` to TFLite path.

## Files

- `tf_dataset.py`: subject-wise K-fold dataset reader for the existing data. The default Keras path returns five inputs: cropRGB, cropIR, RGB, IR, and heatmap.
- `tf_model.py`: 5-input multimodal MobileNetV2 Keras model. RGB streams use ImageNet weights; 1-channel streams can copy those weights by averaging the first RGB convolution to one channel.
- `train_tf.py`: trains and saves `.keras` checkpoints by best validation ACER.
- `convert_h5_to_tflite.py`: converts a saved Keras model to float, standard full INT8, or NPU-friendly full INT8 TFLite.

## Typical commands

Run on the sub-laptop GPU environment. Use the root scripts because they set the
TensorFlow CUDA library path automatically:

```bash
./run_keras_model.sh
./run_keras_train.sh --epochs 30 --folds 5 --fold-idx 0
./run_keras_convert.sh --float --int8 --fold-idx 0
.venv/bin/python keras_pipeline/convert_h5_to_tflite.py --npu-int8 --fold-idx 4 --calibration-samples 500
.venv/bin/python evaluate_tflite.py --models \
  model/keras/best_model_fold0_float.tflite \
  model/keras/best_model_fold0_int8.tflite
```

The generated files go under `model/keras/` by default.

`--npu-int8` writes `model/keras/best_model_fold{N}_npu_int8.tflite`. It reuses the trained `.keras` weights and changes only the export graph:

- removes the RGB normalization Lambda from the TFLite graph,
- exports RGB input in MobileNet `[-1,1]` range,
- replaces `MEAN` global pooling with `AVERAGE_POOL_2D`,
- fixes batch size to 1 for Android deployment.

Android `model_spec.json` must match this export: cropRGB/RGB use MobileNet `[-1,1]` input range, cropIR/IR use `mean=[0.5]`, `std=[0.5]`, and heatmap uses raw `0..1`. The standard float/int8 exports use RGB ImageNet mean/std instead.

Current target-board status: the NPU-friendly export still fails Android NNAPI with `ANEURALNETWORKS_BAD_DATA ... while adding operation` and falls back to CPU/XNNPACK. Treat `Backend CPU` as CPU inference, not NPU acceleration.

Current fold 4 validation snapshot:

```text
best_model_fold4_int8.tflite      val_acc=0.9971 APCER=0.0000 BPCER=0.0120 ACER=0.0060
best_model_fold4_npu_int8.tflite  val_acc=0.9924 APCER=0.0000 BPCER=0.0320 ACER=0.0160
```

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
