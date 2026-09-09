"""High-frequency residual map auxiliary-supervision regression tests."""
import inspect

import cv2
import numpy as np
import pytest
import tensorflow as tf

from keras_pipeline.data.dataset import make_dataset, make_single_dataset
from keras_pipeline.data.depth_generator import generate_high_frequency_residual_map
from keras_pipeline.models.losses import build_classification_loss, build_supcon_loss
from keras_pipeline.models.model import build_dual_model, build_single_model, extract_deploy_model


def test_residual_option_preserves_existing_positional_argument_order():
    single_model_params = list(inspect.signature(build_single_model).parameters)
    dual_model_params = list(inspect.signature(build_dual_model).parameters)
    dual_dataset_params = list(inspect.signature(make_dataset).parameters)
    single_dataset_params = list(inspect.signature(make_single_dataset).parameters)

    expected_model_tail = [
        "aux_depth", "aux_binary_pad", "aux_supcon", "projection_dim", "aux_residual"
    ]
    assert single_model_params[-5:] == expected_model_tail
    assert dual_model_params[-5:] == expected_model_tail
    assert dual_dataset_params[-5:] == [
        "aux_depth", "aux_binary_pad", "aux_supcon", "aux_residual", "augment_display_artifacts"
    ]
    assert single_dataset_params[-5:] == [
        "aux_depth", "aux_binary_pad", "aux_supcon", "aux_residual", "augment_display_artifacts"
    ]


def test_residual_generator_rejects_non_grayscale_input():
    with pytest.raises(ValueError, match="single-channel"):
        generate_high_frequency_residual_map(np.zeros((224, 224, 3), dtype=np.float32))


def test_residual_generator_is_zero_for_constant_and_responds_to_texture():
    constant = np.full((224, 224, 1), 0.5, dtype=np.float32)
    checkerboard = (np.indices((224, 224)).sum(axis=0) % 2).astype(np.float32)[..., np.newaxis]

    constant_map = generate_high_frequency_residual_map(constant)
    texture_map = generate_high_frequency_residual_map(checkerboard)

    assert constant_map.shape == (14, 14, 1)
    assert texture_map.shape == (14, 14, 1)
    assert constant_map.dtype == np.float32
    assert np.all(constant_map == 0.0)
    assert 0.0 <= texture_map.min() <= texture_map.max() <= 1.0
    assert texture_map.mean() > 0.4


def test_residual_head_combines_with_depth_and_supcon_and_is_stripped_for_deploy():
    model = build_single_model(
        input_type="crop_ir",
        rgb_weights=None,
        classifier_units=0,
        aux_depth=True,
        aux_supcon=True,
        aux_residual=True,
        projection_dim=16,
    )
    dummy_input = np.zeros((2, 224, 224, 1), dtype=np.float32)
    outputs = model(dummy_input)

    assert model.output_names == ["logits", "depth_output", "residual_output", "supcon_output"]
    assert [tuple(output.shape) for output in outputs] == [
        (2, 12),
        (2, 14, 14, 1),
        (2, 14, 14, 1),
        (2, 16),
    ]

    deploy_model = extract_deploy_model(model)
    np.testing.assert_allclose(deploy_model(dummy_input), outputs[0], rtol=0.0, atol=0.0)
    assert len(deploy_model.outputs) == 1
    assert all("residual" not in layer.name for layer in deploy_model.layers)


def test_dataset_builds_residual_target_for_every_ir_class(tmp_path):
    image_path = str(tmp_path / "test.bmp")
    checkerboard = ((np.indices((224, 224)).sum(axis=0) % 2) * 255).astype(np.uint8)
    assert cv2.imwrite(image_path, checkerboard)
    items = [(image_path, image_path, 0), (image_path, image_path, 4)]

    dataset = make_single_dataset(
        items,
        input_type="crop_ir",
        batch_size=2,
        aux_depth=True,
        aux_supcon=True,
        aux_residual=True,
    )
    _, targets = next(iter(dataset))

    assert set(targets) == {"logits", "depth_output", "residual_output", "supcon_output"}
    assert targets["residual_output"].shape == (2, 14, 14, 1)
    assert np.all(targets["residual_output"].numpy() > 0.4)
    np.testing.assert_array_equal(
        targets["residual_output"][0].numpy(), targets["residual_output"][1].numpy()
    )


def test_crop_rgb_dataset_rejects_ir_residual_supervision(tmp_path):
    image_path = str(tmp_path / "test.bmp")
    assert cv2.imwrite(image_path, np.zeros((224, 224, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="IR inputs"):
        make_single_dataset(
            [(image_path, image_path, 0)],
            input_type="crop_rgb",
            aux_residual=True,
        )


def test_dual_model_and_dataset_use_ir_residual_target(tmp_path):
    model = build_dual_model(
        rgb_weights=None,
        classifier_units=0,
        aux_residual=True,
    )
    outputs = model([
        np.zeros((1, 224, 224, 3), dtype=np.float32),
        np.zeros((1, 224, 224, 1), dtype=np.float32),
    ])
    assert model.output_names == ["logits", "residual_output"]
    assert outputs[0].shape == (1, 12)
    assert outputs[1].shape == (1, 14, 14, 1)

    rgb_path = str(tmp_path / "rgb.bmp")
    ir_path = str(tmp_path / "ir.bmp")
    assert cv2.imwrite(rgb_path, np.zeros((224, 224, 3), dtype=np.uint8))
    assert cv2.imwrite(ir_path, np.zeros((224, 224), dtype=np.uint8))
    _, targets = next(iter(make_dataset(
        [(rgb_path, ir_path, 0)], batch_size=1, aux_residual=True
    )))
    assert set(targets) == {"logits", "residual_output"}
    assert targets["residual_output"].shape == (1, 14, 14, 1)


def test_depth_supcon_and_residual_heads_train_together_and_reload_for_deploy(tmp_path):
    model = build_single_model(
        input_type="crop_ir",
        rgb_weights=None,
        classifier_units=0,
        dropout=0.0,
        aux_depth=True,
        aux_supcon=True,
        aux_residual=True,
        projection_dim=16,
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
        loss={
            "logits": build_classification_loss(label_smoothing=0.0),
            "depth_output": tf.keras.losses.MeanSquaredError(),
            "residual_output": tf.keras.losses.MeanSquaredError(),
            "supcon_output": build_supcon_loss(temperature=0.1),
        },
        loss_weights={
            "logits": 1.0,
            "depth_output": 0.5,
            "residual_output": 0.1,
            "supcon_output": 0.1,
        },
    )
    inputs = np.random.default_rng(42).random((4, 224, 224, 1), dtype=np.float32)
    targets = {
        "logits": np.array([0, 10, 1, 4], dtype=np.int32),
        "depth_output": np.zeros((4, 14, 14, 1), dtype=np.float32),
        "residual_output": np.random.default_rng(7).random((4, 14, 14, 1), dtype=np.float32),
        "supcon_output": np.array([0, 10, 1, 4], dtype=np.int32),
    }

    result = model.train_on_batch(inputs, targets, return_dict=True)

    assert all(np.isfinite(value) for value in result.values())

    checkpoint = tmp_path / "residual.keras"
    model.save(checkpoint)
    loaded = tf.keras.models.load_model(checkpoint, compile=False)
    deploy_model = extract_deploy_model(loaded)

    assert deploy_model.output_shape == (None, 12)
    assert all("residual" not in layer.name for layer in deploy_model.layers)
