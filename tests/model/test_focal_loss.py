"""keras_pipeline.losses 모듈의 손실 함수 및 Focal Loss 동작을 검증한다."""
import numpy as np
import pytest
import tensorflow as tf

from classes import CLASS_NAMES
from keras_pipeline.losses import build_classification_loss


class TestBuildClassificationLoss:
    def test_invalid_loss_type_raises(self):
        with pytest.raises(ValueError, match="지원하지 않는 손실 함수"):
            build_classification_loss(loss_type="invalid_loss")

    def test_ce_loss_without_smoothing(self):
        loss_fn = build_classification_loss(loss_type="ce", label_smoothing=0.0)
        y_true = tf.constant([0, 1])
        y_pred = tf.constant([[5.0] + [0.0] * 11, [0.0, 5.0] + [0.0] * 10])
        val = loss_fn(y_true, y_pred)
        assert val.shape == ()
        assert val.numpy() > 0

    def test_ce_loss_with_smoothing(self):
        loss_fn = build_classification_loss(loss_type="ce", label_smoothing=0.1)
        y_true = tf.constant([0, 1])
        y_pred = tf.constant([[5.0] + [0.0] * 11, [0.0, 5.0] + [0.0] * 10])
        val = loss_fn(y_true, y_pred)
        assert val.shape == ()
        assert val.numpy() > 0

    def test_focal_loss_suppresses_easy_samples(self):
        """Focal Loss가 이미 잘 맞히는 쉬운 샘플의 손실을 억제하고 어려운 샘플에 집중하는지 검증한다."""
        ce_fn = build_classification_loss(loss_type="ce", label_smoothing=0.0)
        focal_fn = build_classification_loss(loss_type="focal", label_smoothing=0.0, focal_gamma=2.0, focal_alpha=1.0)

        # y_true = 0 (클래스 0)
        # 쉬운 샘플: 클래스 0에 높은 로짓 (p_0 ≈ 0.999)
        y_easy = tf.constant([0])
        pred_easy = tf.constant([[10.0] + [0.0] * 11])

        # 어려운 샘플: 클래스 0인데 다른 클래스에 높은 로짓 (오답, p_0 ≈ 0.001)
        y_hard = tf.constant([0])
        pred_hard = tf.constant([[0.0, 10.0] + [0.0] * 10])

        ce_easy = ce_fn(y_easy, pred_easy).numpy()
        ce_hard = ce_fn(y_hard, pred_hard).numpy()

        focal_easy = focal_fn(y_easy, pred_easy).numpy()
        focal_hard = focal_fn(y_hard, pred_hard).numpy()

        # Focal Loss에서 쉬운 샘플의 손실은 CE 대비 대폭 감소해야 함
        assert focal_easy < ce_easy
        # 난이도별 손실 비율 (Hard / Easy)이 Focal Loss에서 훨씬 극대화되어야 함
        ce_ratio = ce_hard / ce_easy
        focal_ratio = focal_hard / focal_easy
        assert focal_ratio > ce_ratio * 10.0

    def test_focal_loss_train_step_smoke(self):
        """Focal Loss를 적용한 모델이 정상적으로 컴파일 및 1스텝 학습되는지 검증한다."""
        inputs = tf.keras.Input(shape=(16,))
        outputs = tf.keras.layers.Dense(len(CLASS_NAMES))(inputs)
        model = tf.keras.Model(inputs, outputs)

        loss_fn = build_classification_loss(loss_type="focal", label_smoothing=0.1, focal_gamma=2.0)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=loss_fn,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        )

        x = np.random.randn(32, 16).astype(np.float32)
        y = np.random.randint(0, len(CLASS_NAMES), size=(32,)).astype(np.int32)

        history = model.fit(x, y, epochs=1, batch_size=16, verbose=0)
        assert "loss" in history.history
        assert len(history.history["loss"]) == 1
