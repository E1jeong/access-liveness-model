import numpy as np
import pytest
import tensorflow as tf

from keras_pipeline.efficientnet_lite import EfficientNetLite0
from keras_pipeline.tf_model import build_dual_model, build_single_model
from keras_pipeline.export_validator import build_npu_export_model, validate_npu_export_parity


def test_efficientnet_lite_backbone_shapes():
    """Test standard EfficientNetLite0 backbone instantiation and shapes."""
    # 3-channel input with average pooling
    m_rgb = EfficientNetLite0(input_shape=(224, 224, 3), pooling="avg", weights=None)
    assert m_rgb.output_shape == (None, 1280)
    assert m_rgb.count_params() == 3413024

    # 1-channel input with average pooling
    m_ir = EfficientNetLite0(input_shape=(224, 224, 1), pooling="avg", weights=None)
    assert m_ir.output_shape == (None, 1280)
    assert m_ir.count_params() == 3412448


def test_efficientnet_lite_stem_reduction_parity():
    """Mathematical parity test for EfficientNet-Lite0 stem_conv reduction (sum vs mean)."""
    model_sum = build_single_model(
        input_type="crop_ir",
        rgb_weights="imagenet",
        conv1_reduction="sum",
        backbone="efficientnet_lite0",
    )
    model_mean = build_single_model(
        input_type="crop_ir",
        rgb_weights="imagenet",
        conv1_reduction="mean",
        backbone="efficientnet_lite0",
    )
    model_rgb = build_single_model(
        input_type="crop_rgb",
        rgb_weights="imagenet",
        backbone="efficientnet_lite0",
    )

    np.random.seed(42)
    x_gray = np.random.uniform(-1.0, 1.0, size=(1, 224, 224, 1)).astype(np.float32)
    x_rgb = np.concatenate([x_gray, x_gray, x_gray], axis=-1)

    rgb_backbone = model_rgb.get_layer("crop_rgb_efficientnet_lite0")
    ir_sum_backbone = model_sum.get_layer("crop_ir_efficientnet_lite0")
    ir_mean_backbone = model_mean.get_layer("crop_ir_efficientnet_lite0")

    stem_rgb = rgb_backbone.get_layer("stem_conv")
    stem_sum = ir_sum_backbone.get_layer("stem_conv")
    stem_mean = ir_mean_backbone.get_layer("stem_conv")

    # Pad layers
    pad_rgb = rgb_backbone.get_layer("stem_conv_pad")
    pad_ir = ir_sum_backbone.get_layer("stem_conv_pad")

    out_rgb = stem_rgb(pad_rgb(x_rgb)).numpy()
    out_sum = stem_sum(pad_ir(x_gray)).numpy()
    out_mean = stem_mean(pad_ir(x_gray)).numpy()

    # Sum reduction output must exactly equal RGB output on grayscale input
    max_sum_diff = np.max(np.abs(out_rgb - out_sum))
    assert max_sum_diff < 1e-5, f"Sum reduction mismatch: {max_sum_diff}"

    # Mean reduction output must equal 1/3 of RGB output
    max_mean_diff = np.max(np.abs(out_rgb / 3.0 - out_mean))
    assert max_mean_diff < 1e-5, f"Mean reduction mismatch: {max_mean_diff}"

    # Sum == 3 * Mean
    max_scale_diff = np.max(np.abs(out_sum - 3.0 * out_mean))
    assert max_scale_diff < 1e-5, f"Sum vs 3*Mean scaling mismatch: {max_scale_diff}"


def test_efficientnet_lite_dual_and_single_models():
    """Verify dual, crop_rgb, and crop_ir model builders with efficientnet_lite0 backbone."""
    # Dual model
    dual_model = build_dual_model(backbone="efficientnet_lite0", rgb_weights=None)
    assert len(dual_model.inputs) == 2
    assert tuple(dual_model.inputs[0].shape) == (None, 224, 224, 3)
    assert tuple(dual_model.inputs[1].shape) == (None, 224, 224, 1)
    assert dual_model.output_shape == (None, 10)

    # Single crop_rgb
    rgb_model = build_single_model(input_type="crop_rgb", backbone="efficientnet_lite0", rgb_weights=None)
    assert len(rgb_model.inputs) == 1
    assert tuple(rgb_model.inputs[0].shape) == (None, 224, 224, 3)
    assert rgb_model.output_shape == (None, 10)

    # Single crop_ir
    ir_model = build_single_model(input_type="crop_ir", backbone="efficientnet_lite0", rgb_weights=None)
    assert len(ir_model.inputs) == 1
    assert tuple(ir_model.inputs[0].shape) == (None, 224, 224, 1)
    assert ir_model.output_shape == (None, 10)


def test_efficientnet_lite_npu_export_parity():
    """Verify NPU export graph reconstruction and parity for EfficientNet-Lite0."""
    # 1. Dual parity
    dual_train = build_dual_model(backbone="efficientnet_lite0", rgb_weights=None)
    dual_export = build_npu_export_model(dual_train, "dual")
    dual_sample = [
        np.random.uniform(-1.0, 1.0, size=(224, 224, 3)).astype(np.float32),
        np.random.uniform(-1.0, 1.0, size=(224, 224, 1)).astype(np.float32),
    ]
    dual_parity = validate_npu_export_parity(dual_train, dual_export, dual_sample, "dual")
    assert dual_parity["argmax_agreement"] is True
    assert dual_parity["max_error"] < 1e-5

    # 2. Crop IR parity
    ir_train = build_single_model(input_type="crop_ir", backbone="efficientnet_lite0", rgb_weights=None)
    ir_export = build_npu_export_model(ir_train, "crop_ir")
    ir_sample = [
        np.random.uniform(-1.0, 1.0, size=(224, 224, 3)).astype(np.float32),
        np.random.uniform(-1.0, 1.0, size=(224, 224, 1)).astype(np.float32),
    ]
    ir_parity = validate_npu_export_parity(ir_train, ir_export, ir_sample, "crop_ir")
    assert ir_parity["argmax_agreement"] is True
    assert ir_parity["max_error"] < 1e-5
