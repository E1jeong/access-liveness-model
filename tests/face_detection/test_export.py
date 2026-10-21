import numpy as np
import pytest
import torch

from face_detection_pipeline.export.convert import (
    ExportModel,
    load_checkpoint,
    unpack_predictions,
    validate_output_dir,
    validate_tflite_contract,
    _predict,
)
from face_detection_pipeline.models.yunet import build_yunet_n
from face_detection_pipeline.training.train import UPSTREAM_COMMIT


@pytest.mark.parametrize("width,height", [(320, 320), (256, 448)])
def test_export_pack_preserves_every_head_and_level(width, height):
    model = build_yunet_n().eval()
    image = torch.rand(1, 3, height, width) * 255
    with torch.inference_mode():
        original = model(image)
        logits, boxes = ExportModel(model)(image)
        assert logits.shape[-1] == 2
        assert boxes.shape[-1] == 4
        packed = torch.cat([logits, boxes], dim=-1)
    restored = unpack_predictions(packed, (width, height))
    assert packed.shape == (1, sum((width // s) * (height // s) for s in (8, 16, 32)), 6)
    for original_group, restored_group in zip(original, restored):
        for expected, actual in zip(original_group, restored_group):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_checkpoint_rejects_incompatible_model_and_source(tmp_path):
    path = tmp_path / "checkpoint.pth"
    for values in [
        {"model": "pad"},
        {"model": "yunet_n_box_only", "upstream_commit": "different"},
        {"model": "yunet_n_box_only", "upstream_commit": UPSTREAM_COMMIT, "input_size": [321, 320]},
    ]:
        torch.save(values, path)
        with pytest.raises(ValueError):
            load_checkpoint(path)


def test_checkpoint_roundtrip_preserves_trained_weights_and_identity(tmp_path):
    path = tmp_path / "checkpoint.pth"
    model = build_yunet_n().eval()
    torch.save({"model": "yunet_n_box_only", "upstream_commit": UPSTREAM_COMMIT,
                "input_size": [320, 320], "run_id": "test", "epoch": 3,
                "model_state_dict": model.state_dict()}, path)
    restored, checkpoint, digest = load_checkpoint(path)
    assert checkpoint["epoch"] == 3
    assert len(digest) == 64
    for key, expected in model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[key], expected, rtol=0, atol=0)


def test_int8_prediction_quantizes_rgb_and_dequantizes_output():
    class Interpreter:
        def get_input_details(self):
            return [{"index": 0, "dtype": np.int8, "quantization": (1.0, -128)}]

        def get_output_details(self):
            # Deliberately reversed tensor order and different scales.
            return [{"index": 1, "shape": [1, 1, 4], "dtype": np.int8, "quantization": (0.25, -10)},
                    {"index": 2, "shape": [1, 1, 2], "dtype": np.int8, "quantization": (0.5, -20)}]

        def set_tensor(self, index, values):
            np.testing.assert_array_equal(values, np.array([[[[-128, 0, 127]]]], dtype=np.int8))

        def invoke(self):
            pass

        def get_tensor(self, index):
            return np.array([[[2, 6, -14, -18]]] if index == 1 else [[[-20, -18]]], dtype=np.int8)

    image = torch.tensor([[[0]], [[128]], [[255]]], dtype=torch.float32)
    result = _predict(Interpreter(), image)
    np.testing.assert_array_equal(result, [[[0, 1, 3, 4, -1, -2]]])


def test_output_dir_rejects_overwrite_and_paths_outside_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="model/"):
        validate_output_dir(tmp_path / "outside")
    existing = tmp_path / "model" / "existing"
    existing.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        validate_output_dir(existing)
    validate_output_dir(tmp_path / "model" / "new")


def test_tflite_contract_rejects_wrong_shape_dtype_and_unscaled_int8():
    inputs = [{"shape": np.array([1, 320, 320, 3]), "dtype": np.float32}]
    outputs = [{"shape": np.array([1, 2100, 2]), "dtype": np.float32},
               {"shape": np.array([1, 2100, 4]), "dtype": np.float32}]
    validate_tflite_contract(inputs, outputs, (320, 320), False)
    for bad in [
        {"shape": np.array([1, 12]), "dtype": np.float32},
        {"shape": np.array([1, 6, 2100]), "dtype": np.float32},
        {"shape": np.array([1, 2100, 6]), "dtype": np.int8},
    ]:
        with pytest.raises(ValueError):
            validate_tflite_contract(inputs, [bad, outputs[1]], (320, 320), False)
    int_inputs = [dict(inputs[0], dtype=np.int8, quantization=(0.0, 0))]
    int_outputs = [dict(info, dtype=np.int8, quantization=(0.1, 0)) for info in outputs]
    with pytest.raises(ValueError, match="scale"):
        validate_tflite_contract(int_inputs, int_outputs, (320, 320), True)


@pytest.mark.parametrize("width,height", [(320, 320), (256, 448)])
def test_landmark_export_preserves_all_heads(width, height):
    model = build_yunet_n(landmarks=True).eval()
    image = torch.rand(1, 3, height, width) * 255
    with torch.inference_mode():
        original = model(image)
        parts = ExportModel(model)(image)
    assert [part.shape[-1] for part in parts] == [2, 4, 10]
    restored = unpack_predictions(torch.cat(parts, dim=-1), (width, height))
    for expected_group, actual_group in zip(original, restored):
        for expected, actual in zip(expected_group, actual_group):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_landmark_checkpoint_contract_and_roundtrip(tmp_path):
    from face_detection_pipeline.data.dataset import LANDMARK_NAMES

    path = tmp_path / "landmarks.pth"
    checkpoint = {"model": "yunet_n_landmarks", "upstream_commit": UPSTREAM_COMMIT,
                  "input_size": [320, 320], "run_id": "landmarks", "epoch": 25,
                  "model_state_dict": build_yunet_n(landmarks=True).state_dict(),
                  "landmark_names": list(LANDMARK_NAMES),
                  "landmark_encoding": "grid_relative_stride_units_xy"}
    torch.save(checkpoint, path)
    model, restored, digest = load_checkpoint(path)
    assert model.model.landmarks
    assert restored["epoch"] == 25
    assert len(digest) == 64
    checkpoint["landmark_names"] = list(reversed(LANDMARK_NAMES))
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="landmark"):
        load_checkpoint(path)


def test_landmark_tflite_contract_requires_separate_scaled_third_tensor():
    inputs = [{"shape": np.array([1, 320, 320, 3]), "dtype": np.int8, "quantization": (1., -128)}]
    outputs = [{"shape": np.array([1, 2100, channels]), "dtype": np.int8,
                "quantization": (0.1, 0)} for channels in (2, 4, 10)]
    validate_tflite_contract(inputs, outputs[::-1], (320, 320), True, landmarks=True)
    for bad in [outputs[:2], outputs + outputs[-1:],
                outputs[:2] + [dict(outputs[2], quantization=(0., 0))]]:
        with pytest.raises(ValueError):
            validate_tflite_contract(inputs, bad, (320, 320), True, landmarks=True)
    with pytest.raises(ValueError):
        validate_tflite_contract(inputs, outputs, (320, 320), True)
