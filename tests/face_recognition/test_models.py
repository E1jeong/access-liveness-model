import os
from pathlib import Path
import h5py
import numpy as np
import pytest
from face_recognition_pipeline.models.mobilenet_v1 import (
    MIN_EXPECTED_LOADED_LAYERS,
    build_npu_mobilenet_model,
    load_mobilenet_h5_weights,
)


pytestmark = pytest.mark.tensorflow


def test_build_npu_mobilenet_architecture():
    """Verify NPU-compliant layer contracts without requiring pre-trained weights."""
    model = build_npu_mobilenet_model(weights_path=None, embedding_dim=256)

    # 1. Fixed batch shape
    assert model.input_shape == (1, 112, 112, 3)
    assert model.output_shape == (1, 256)

    # 2. Layer name contracts
    layer_names = [layer.name for layer in model.layers]
    assert "depthwise_conv2d" in layer_names
    assert "batch_normalization" in layer_names
    assert "conv2d" in layer_names
    assert "embedding" in layer_names
    assert "out_embedding" in layer_names

    # 3. Forward pass sanity check
    dummy_input = np.zeros((1, 112, 112, 3), dtype=np.float32)
    output = model(dummy_input, training=False)
    assert output.shape == (1, 256)


def test_load_weights_raises_on_missing_file():
    model = build_npu_mobilenet_model(weights_path=None)
    with pytest.raises(FileNotFoundError, match="not found"):
        load_mobilenet_h5_weights(model, "non_existent_weights.h5")


def test_load_weights_raises_on_insufficient_layer_mapping(tmp_path):
    empty_h5 = tmp_path / "empty_weights.h5"
    with h5py.File(empty_h5, "w") as f:
        f.create_group("model_weights")

    model = build_npu_mobilenet_model(weights_path=None)
    with pytest.raises(RuntimeError, match="Insufficient weights loaded"):
        load_mobilenet_h5_weights(model, str(empty_h5))


def test_load_production_weights_if_available():
    repo_root = Path(__file__).resolve().parents[2]
    weights_path = repo_root / "model" / "recognition" / "mobilenet_emb256.h5"
    if not weights_path.exists():
        pytest.skip("Production weights not present in test environment.")

    model = build_npu_mobilenet_model(weights_path=str(weights_path))
    dummy_input = np.zeros((1, 112, 112, 3), dtype=np.float32)
    output = model(dummy_input, training=False)
    assert output.shape == (1, 256)
    assert not np.isnan(output.numpy()).any()
