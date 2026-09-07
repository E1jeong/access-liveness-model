"""Supervised Contrastive Learning 보조 학습 회귀 테스트."""
import numpy as np
import tensorflow as tf

from keras_pipeline.data.dataset import make_single_dataset
from keras_pipeline.models.losses import build_classification_loss, build_supcon_loss
from keras_pipeline.models.model import build_single_model, extract_deploy_model


def test_supcon_loss_pulls_bona_fide_and_uses_spoof_as_negatives():
    loss_fn = build_supcon_loss(temperature=0.1)
    labels = tf.constant([0, 10, 11, 1], dtype=tf.int32)
    clustered = tf.constant(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=tf.float32
    )
    mixed = tf.constant(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]], dtype=tf.float32
    )

    assert float(loss_fn(labels, clustered)) < float(loss_fn(labels, mixed))


def test_supcon_loss_is_finite_without_a_positive_pair():
    loss_fn = build_supcon_loss(temperature=0.1)
    loss = loss_fn(
        tf.constant([0, 1, 2], dtype=tf.int32),
        tf.constant([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=tf.float32),
    )

    assert np.isfinite(float(loss))
    assert float(loss) == 0.0


def test_supcon_batch_32_has_finite_gradients():
    labels = tf.constant(([0, 10, 11, 0] + list(range(1, 10)) * 4)[:32], dtype=tf.int32)
    embeddings = tf.Variable(tf.random.stateless_normal((32, 128), seed=(4, 2)))

    with tf.GradientTape() as tape:
        loss = build_supcon_loss(temperature=0.1)(labels, embeddings)
    gradients = tape.gradient(loss, embeddings)

    assert np.isfinite(float(loss))
    assert np.all(np.isfinite(gradients.numpy()))
    assert np.any(gradients.numpy() != 0.0)


def test_supcon_projection_head_is_training_only():
    model = build_single_model(
        input_type="crop_ir", rgb_weights=None, classifier_units=0,
        aux_supcon=True, projection_dim=16,
    )
    dummy_input = np.random.default_rng(42).random((2, 224, 224, 1), dtype=np.float32)
    outputs = model(dummy_input)

    assert len(outputs) == 2
    assert model.output_names == ["logits", "supcon_output"]
    assert outputs[0].shape == (2, 12)
    assert outputs[1].shape == (2, 16)
    np.testing.assert_allclose(
        tf.norm(outputs[1], axis=-1).numpy(), np.ones(2), rtol=1e-5, atol=1e-5
    )

    deploy_model = extract_deploy_model(model)
    assert len(deploy_model.outputs) == 1
    assert deploy_model(dummy_input).shape == (2, 12)
    assert all("supcon" not in layer.name for layer in deploy_model.layers)


def test_supcon_model_trains_and_checkpoint_loads_for_deploy(tmp_path):
    model = build_single_model(
        input_type="crop_ir", rgb_weights=None, classifier_units=0, dropout=0.0,
        aux_supcon=True, projection_dim=16,
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
        loss={
            "logits": build_classification_loss(label_smoothing=0.0),
            "supcon_output": build_supcon_loss(temperature=0.1),
        },
        loss_weights={"logits": 1.0, "supcon_output": 0.1},
    )
    inputs = np.random.default_rng(7).random((4, 224, 224, 1), dtype=np.float32)
    labels = np.array([0, 10, 1, 2], dtype=np.int32)
    result = model.train_on_batch(
        inputs, {"logits": labels, "supcon_output": labels}, return_dict=True
    )

    assert all(np.isfinite(value) for value in result.values())

    checkpoint = tmp_path / "supcon.keras"
    model.save(checkpoint)
    loaded = tf.keras.models.load_model(checkpoint, compile=False)
    deploy_model = extract_deploy_model(loaded)
    assert deploy_model.output_shape == (None, 12)
    assert all("supcon" not in layer.name for layer in deploy_model.layers)


def test_dataset_reuses_multiclass_label_for_supcon(tmp_path):
    import cv2

    image_path = str(tmp_path / "test.bmp")
    assert cv2.imwrite(image_path, np.zeros((224, 224), dtype=np.uint8))
    items = [(image_path, image_path, 0), (image_path, image_path, 1)]
    dataset = make_single_dataset(
        items, input_type="crop_ir", batch_size=2, aux_supcon=True,
    )

    _, targets = next(iter(dataset))
    assert set(targets) == {"logits", "supcon_output"}
    np.testing.assert_array_equal(targets["logits"].numpy(), targets["supcon_output"].numpy())
