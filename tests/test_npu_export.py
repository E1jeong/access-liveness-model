import numpy as np
import pytest
from tensorflow import keras

from keras_pipeline.convert_keras_to_tflite import (
    _copy_nested_weights,
    build_npu_export_model,
    validate_npu_export_parity,
)
from keras_pipeline.tf_dataset import RGB_MEAN, RGB_STD
from keras_pipeline.tf_model import build_single_mobilenetv2


def _nested_model(weighted_layer_name, units, extra_weighted_layer_name=None):
    inputs = keras.Input(shape=(4,))
    backbone_layers = [keras.Input(shape=(4,)), keras.layers.Dense(units, name=weighted_layer_name)]
    if extra_weighted_layer_name:
        backbone_layers.append(keras.layers.Dense(units, name=extra_weighted_layer_name))
    backbone = keras.Sequential(
        backbone_layers,
        name="backbone",
    )
    return keras.Model(inputs, backbone(inputs))


def test_copy_nested_weights_rejects_missing_weighted_layer():
    source = _nested_model("source_only", 3)
    target = _nested_model("target_only", 3)

    with pytest.raises(ValueError, match="export backbone에 없는 source weighted layer"):
        _copy_nested_weights(source, target, "backbone")


def test_copy_nested_weights_rejects_export_only_weighted_layer():
    source = _nested_model("shared", 3)
    target = _nested_model("shared", 3, extra_weighted_layer_name="export_only")

    with pytest.raises(ValueError, match="source backbone에 없는 export weighted layer"):
        _copy_nested_weights(source, target, "backbone")


def test_copy_nested_weights_rejects_weight_shape_mismatch():
    source = _nested_model("shared", 3)
    target = _nested_model("shared", 4)

    with pytest.raises(ValueError, match="weight tensor shape mismatch"):
        _copy_nested_weights(source, target, "backbone")


def test_npu_export_preserves_crop_rgb_logits():
    trained_model = build_single_mobilenetv2(
        input_type="crop_rgb",
        rgb_weights=None,
        dropout=0.0,
        classifier_units=8,
        gray_imagenet_init=False,
    )
    export_model = build_npu_export_model(trained_model, "crop_rgb")
    raw_rgb = np.random.default_rng(42).random((224, 224, 3), dtype=np.float32)
    normalized_rgb = (raw_rgb - RGB_MEAN) / RGB_STD

    result = validate_npu_export_parity(
        trained_model, export_model, (normalized_rgb, None), "crop_rgb"
    )

    assert result["max_error"] < 1e-5
    assert result["mean_error"] < 1e-5
    assert result["argmax_agreement"]
