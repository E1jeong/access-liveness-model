import json

import pytest

from common.evaluate_tflite import _read_npu_int8_contract


def _write_manifest(tflite_path, inputs):
    manifest_path = str(tflite_path).replace(".tflite", "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"model_type": "crop_rgb", "inputs": inputs}, f)


RGB_NPU_INPUT = {
    "name": "rgb",
    "input_type": "rgb",
    "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "range": "[-1, 1]"},
}
RGB_IMAGENET_INPUT = {
    "name": "rgb",
    "input_type": "rgb",
    "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "range": "imagenet"},
}
IR_INPUT = {
    "name": "ir",
    "input_type": "ir",
    "normalization": {"mean": [0.5], "std": [0.5], "range": "[-1, 1]"},
}


def test_rgb_imagenet_contract_is_not_npu_int8(tmp_path):
    model_path = tmp_path / "model_int8.tflite"
    _write_manifest(model_path, [RGB_IMAGENET_INPUT])
    assert _read_npu_int8_contract(str(model_path)) is False


def test_rgb_minus_one_to_one_contract_is_npu_int8(tmp_path):
    model_path = tmp_path / "model.tflite"
    _write_manifest(model_path, [RGB_NPU_INPUT])
    assert _read_npu_int8_contract(str(model_path)) is True


def test_filename_does_not_override_manifest(tmp_path):
    # basename에 "npu_int8"이 있어도 manifest가 imagenet 계약이면 False여야 한다.
    model_path = tmp_path / "model_npu_int8.tflite"
    _write_manifest(model_path, [RGB_IMAGENET_INPUT])
    assert _read_npu_int8_contract(str(model_path)) is False


def test_dual_manifest_uses_rgb_input(tmp_path):
    model_path = tmp_path / "dual.tflite"
    _write_manifest(model_path, [RGB_NPU_INPUT, IR_INPUT])
    assert _read_npu_int8_contract(str(model_path)) is True


def test_ir_only_manifest_is_not_npu_int8(tmp_path):
    model_path = tmp_path / "model_npu_int8.tflite"
    _write_manifest(model_path, [IR_INPUT])
    assert _read_npu_int8_contract(str(model_path)) is False


def test_missing_manifest_fails_fast(tmp_path):
    model_path = tmp_path / "model_npu_int8.tflite"
    with pytest.raises(FileNotFoundError):
        _read_npu_int8_contract(str(model_path))
