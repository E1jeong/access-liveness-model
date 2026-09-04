"""Phase 2 binary PAD 보조 지도학습 단위 테스트."""
import numpy as np
import pytest
import tensorflow as tf

from classes import BONA_FIDE_CLASS_INDICES
from keras_pipeline.losses import build_binary_pad_loss
from keras_pipeline.tf_dataset import make_single_dataset
from keras_pipeline.tf_model import build_single_model, extract_deploy_model


def test_phase_two_bona_fide_indices():
    assert BONA_FIDE_CLASS_INDICES == (0, 10, 11)


def test_binary_pad_loss_matches_phase_two_policy():
    loss_fn = build_binary_pad_loss()
    labels = tf.constant([0, 10, 11, 1, 9], dtype=tf.int32)
    correct_logits = tf.constant([[-8.0], [-8.0], [-8.0], [8.0], [8.0]], dtype=tf.float32)
    inverted_logits = -correct_logits

    assert float(loss_fn(labels, correct_logits)) < 0.01
    assert float(loss_fn(labels, inverted_logits)) > 7.0


def test_binary_pad_head_is_training_only_and_deploy_extracts_logits():
    model = build_single_model(
        input_type="crop_ir", rgb_weights=None, aux_depth=True, aux_binary_pad=True,
    )
    outputs = model(np.zeros((2, 224, 224, 1), dtype=np.float32))

    assert len(outputs) == 3
    assert outputs[0].shape == (2, 12)
    assert outputs[1].shape == (2, 14, 14, 1)
    assert outputs[2].shape == (2, 1)

    deploy_model = extract_deploy_model(model)
    assert len(deploy_model.outputs) == 1
    assert deploy_model(np.zeros((2, 224, 224, 1), dtype=np.float32)).shape == (2, 12)


def test_dataset_reuses_multiclass_label_for_binary_pad(tmp_path):
    import cv2

    image_path = str(tmp_path / "test.bmp")
    assert cv2.imwrite(image_path, np.zeros((224, 224), dtype=np.uint8))
    items = [(image_path, image_path, 0), (image_path, image_path, 10)]
    dataset = make_single_dataset(
        items, input_type="crop_ir", batch_size=2, aux_binary_pad=True,
    )

    _, targets = next(iter(dataset))
    assert set(targets) == {"logits", "pad_output"}
    np.testing.assert_array_equal(targets["logits"].numpy(), targets["pad_output"].numpy())
