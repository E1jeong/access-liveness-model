import json

import tensorflow as tf
from tensorflow import keras

from keras_pipeline.data import spec
from keras_pipeline.export import validator


def _write_tiny_tflite(path, channels=3):
    inputs = keras.Input(shape=(8, 8, channels), name="a_input")
    model = keras.Model(inputs, keras.layers.Flatten(name="flatten")(inputs))
    tflite_bytes = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    path.write_bytes(tflite_bytes)


def _read_manifest(tflite_path):
    manifest_path = str(tflite_path).replace(".tflite", "_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_sidecar_manifest_uses_spec_crop_margin_ratio(monkeypatch, tmp_path):
    tflite_path = tmp_path / "model_fixed_int8.tflite"
    _write_tiny_tflite(tflite_path)
    assert validator.CROP_MARGIN_RATIO == spec.CROP_MARGIN_RATIO
    sentinel_margin_ratio = 0.123456
    monkeypatch.setattr(validator, "CROP_MARGIN_RATIO", sentinel_margin_ratio)

    validator.write_tflite_sidecar_manifest(str(tflite_path), "crop_rgb", npu_int8=False)

    manifest = _read_manifest(tflite_path)
    assert manifest["crop_margin_ratio"] == sentinel_margin_ratio


def test_sidecar_normalization_follows_explicit_flag_not_filename(tmp_path):
    # basename에 "npu_int8"이 없는 경로라도 명시적 True면 NPU 계약이어야 한다.
    npu_path = tmp_path / "model_fixed_int8.tflite"
    _write_tiny_tflite(npu_path)

    validator.write_tflite_sidecar_manifest(str(npu_path), "crop_rgb", npu_int8=True)

    npu_manifest = _read_manifest(npu_path)
    assert npu_manifest["delegate"] == "nnapi"
    assert npu_manifest["inputs"][0]["normalization"] == {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "range": "[-1, 1]",
    }


def test_sidecar_normalization_ignores_npu_int8_in_filename(tmp_path):
    # 역으로 basename에 "npu_int8"이 있어도 명시적 False면 float/int8 계약이어야 한다.
    float_path = tmp_path / "model_npu_int8.tflite"
    _write_tiny_tflite(float_path)

    validator.write_tflite_sidecar_manifest(str(float_path), "crop_rgb", npu_int8=False)

    float_manifest = _read_manifest(float_path)
    assert float_manifest["delegate"] == "cpu"
    assert float_manifest["inputs"][0]["normalization"]["range"] == "imagenet"


def test_experimental_sidecar_preserves_all_twenty_four_outputs(tmp_path):
    from common.classes import CLASS_NAMES

    inputs = keras.Input(shape=(8, 8, 1), batch_size=1, name="b_crop_ir")
    pooled = keras.layers.GlobalAveragePooling2D()(inputs)
    model = keras.Model(inputs, keras.layers.Dense(24, name="logits")(pooled))
    tflite_path = tmp_path / "outdoor_24.tflite"
    tflite_path.write_bytes(tf.lite.TFLiteConverter.from_keras_model(model).convert())
    validator.write_tflite_sidecar_manifest(str(tflite_path), "crop_ir", npu_int8=False)
    manifest = _read_manifest(tflite_path)
    assert manifest["outputs"][0]["shape"] == [1, 24]
    assert manifest["class_order"] == CLASS_NAMES
    assert manifest["outputs"][0]["output_is_logits"] is True
