from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from face_recognition_pipeline.export.verifier import (
    EXPECTED_INPUT_SHAPE,
    EXPECTED_INT8_OPERATOR_COUNT,
    EXPECTED_OUTPUT_SHAPE,
    verify_int8_operator_graph,
)


pytestmark = pytest.mark.tensorflow


def test_constants():
    assert EXPECTED_INT8_OPERATOR_COUNT == 31
    assert EXPECTED_INPUT_SHAPE == (1, 112, 112, 3)
    assert EXPECTED_OUTPUT_SHAPE == (1, 256)


def test_verify_int8_operator_graph_rejects_corrupted_bytes():
    with pytest.raises(Exception):
        verify_int8_operator_graph(b"invalid_tflite_bytes")


def test_verify_int8_operator_graph_rejects_float_model():
    """Verify that a float32 model is rejected by the INT8 contract."""
    inputs = keras.Input(batch_shape=(1, 112, 112, 3))
    x = keras.layers.Conv2D(16, (3, 3), padding="same")(inputs)
    out = keras.layers.Reshape((16 * 112 * 112,))(x)
    model = keras.Model(inputs=inputs, outputs=out)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_tflite = converter.convert()

    with pytest.raises(RuntimeError, match="Recognition INT8 export"):
        verify_int8_operator_graph(float_tflite)


def test_verify_int8_operator_graph_rejects_shape_mismatch():
    """Verify that models with mismatched input or output shapes are rejected."""
    # Model with wrong output shape (128 instead of 256)
    inputs = keras.Input(batch_shape=(1, 112, 112, 3))
    out = keras.layers.Dense(128)(inputs)
    model = keras.Model(inputs=inputs, outputs=out)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()

    with pytest.raises(RuntimeError, match="shape contract failed"):
        verify_int8_operator_graph(tflite_bytes)


def test_verify_production_int8_model_passes():
    repo_root = Path(__file__).resolve().parents[2]
    tflite_path = (
        repo_root
        / "model"
        / "recognition"
        / "tflite"
        / "mobilenet_emore_npu_int8.tflite"
    )
    if not tflite_path.exists():
        pytest.skip("Production TFLite artifact not present in test environment.")

    with open(tflite_path, "rb") as f:
        tflite_bytes = f.read()

    # Should not raise any exception
    verify_int8_operator_graph(tflite_bytes)
