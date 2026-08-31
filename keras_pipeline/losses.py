"""손실 함수 빌더 모듈 (CrossEntropy, Focal Loss 등)."""
from typing import Callable, Optional
import tensorflow as tf

from classes import CLASS_NAMES


def build_classification_loss(
    loss_type: str = "ce",
    label_smoothing: float = 0.1,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    num_classes: int = len(CLASS_NAMES),
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """분류 손실 함수를 생성한다.

    Args:
        loss_type: 'ce' (CrossEntropy) 또는 'focal' (CategoricalFocalCrossentropy).
        label_smoothing: 라벨 스무딩 계수 (0이면 미적용).
        focal_gamma: Focal Loss 감마 파라미터 (어려운 샘플 가중치 지수).
        focal_alpha: Focal Loss 알파 파라미터 (스케일링 계수).
        num_classes: 전체 클래스 수 (기본값: len(CLASS_NAMES)).

    Returns:
        Keras 컴파일에 사용 가능한 loss_fn(y_true, y_pred).
    """
    if loss_type == "focal":
        core_loss = tf.keras.losses.CategoricalFocalCrossentropy(
            alpha=focal_alpha,
            gamma=focal_gamma,
            from_logits=True,
            label_smoothing=label_smoothing,
        )
    elif loss_type == "ce":
        if label_smoothing > 0:
            core_loss = tf.keras.losses.CategoricalCrossentropy(
                from_logits=True,
                label_smoothing=label_smoothing,
            )
        else:
            return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        raise ValueError(f"지원하지 않는 손실 함수 종류: {loss_type} ('ce' 또는 'focal' 사용 가능)")

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_oh = tf.one_hot(y_true_int, depth=num_classes)
        return core_loss(y_true_oh, y_pred)

    return loss_fn
