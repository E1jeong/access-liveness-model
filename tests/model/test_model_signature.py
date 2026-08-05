import pytest

from classes import CLASS_NAMES
from keras_pipeline.model_signature import (
    MODEL_INPUT_SIGNATURES,
    validate_keras_model_signature,
    validate_tflite_model_signature,
)


def test_class_order_matches_android_collection_names():
    assert CLASS_NAMES == [
        "live",
        "print",
        "picture",
        "mask",
        "display",
        "pmask",
        "curved_print",
        "curved_mask",
        "curved_picture",
        "curved_pmask",
    ]


class FakeTensor:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class FakeModel:
    def __init__(self, inputs, output_shape=(None, len(CLASS_NAMES))):
        self.inputs = [FakeTensor(name, (None, *shape)) for name, shape in inputs]
        self.outputs = [FakeTensor("logits", output_shape)]


@pytest.mark.parametrize("model_type", MODEL_INPUT_SIGNATURES)
def test_valid_keras_signatures(model_type):
    validate_keras_model_signature(FakeModel(MODEL_INPUT_SIGNATURES[model_type]), model_type)


def test_keras_signature_rejects_wrong_input_channel():
    with pytest.raises(ValueError, match="b_ir.*shape"):
        validate_keras_model_signature(
            FakeModel((("a_rgb", (224, 224, 3)), ("b_ir", (224, 224, 3)))), "dual"
        )


def test_keras_signature_rejects_wrong_output_classes():
    with pytest.raises(ValueError, match="출력 shape"):
        validate_keras_model_signature(FakeModel(MODEL_INPUT_SIGNATURES["crop_rgb"], (None, 5)), "crop_rgb")


def _tflite_details(model_type):
    inputs = [
        {"name": f"serving_default_{name}:0", "shape": (1, *shape)}
        for name, shape in MODEL_INPUT_SIGNATURES[model_type]
    ]
    return inputs, [{"name": "StatefulPartitionedCall:0", "shape": (1, len(CLASS_NAMES))}]


def test_valid_tflite_signature_accepts_serving_default_names():
    inputs, outputs = _tflite_details("dual")

    validate_tflite_model_signature(inputs, outputs, "dual")


def test_tflite_signature_rejects_wrong_logical_input_name():
    inputs, outputs = _tflite_details("crop_ir")
    inputs[0]["name"] = "serving_default_a_crop_rgb:0"

    with pytest.raises(ValueError, match="허용되지 않은 입력 이름"):
        validate_tflite_model_signature(inputs, outputs, "crop_ir")
