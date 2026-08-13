import numpy as np
import pytest
from tensorflow.keras import layers

from keras_pipeline.export_validator import build_npu_export_model, validate_npu_export_parity
from keras_pipeline.mobilefacenet import MobileFaceNet
from keras_pipeline.tf_model import build_dual_model, build_single_model


def test_mobilefacenet_backbone_has_224_ir_contract_and_gdconv():
    model = MobileFaceNet(weights=None, pooling="avg")
    feature_model = MobileFaceNet(weights=None)

    assert model.input_shape == (None, 224, 224, 1)
    assert model.output_shape == (None, 128)
    assert feature_model.output_shape == (None, 1, 1, 128)
    assert model.get_layer("gdconv").kernel_size == (7, 7)
    assert 900_000 < model.count_params() < 1_300_000
    assert not any(isinstance(layer, layers.PReLU) for layer in model.layers)


def test_mobilefacenet_rejects_pretrained_or_non_ir_inputs():
    with pytest.raises(ValueError, match="pretrained"):
        MobileFaceNet(weights="imagenet")
    with pytest.raises(ValueError, match="1채널"):
        MobileFaceNet(input_shape=(224, 224, 3), weights=None)
    with pytest.raises(ValueError, match="crop_ir"):
        build_single_model(input_type="crop_rgb", backbone="mobilefacenet", rgb_weights=None)
    with pytest.raises(ValueError, match="crop_ir"):
        build_dual_model(backbone="mobilefacenet", rgb_weights=None)


def test_mobilefacenet_crop_ir_npu_export_preserves_logits():
    trained_model = build_single_model(
        input_type="crop_ir",
        backbone="mobilefacenet",
        rgb_weights=None,
        dropout=0.0,
        classifier_units=8,
    )
    export_model = build_npu_export_model(trained_model, "crop_ir")
    sample = (None, np.random.default_rng(42).uniform(-1.0, 1.0, (224, 224, 1)).astype(np.float32))

    result = validate_npu_export_parity(trained_model, export_model, sample, "crop_ir")

    assert result["argmax_agreement"]
    assert result["max_error"] < 1e-5
