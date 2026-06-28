import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classes import CLASS_NAMES


def _rgb_current_norm_to_mobilenet_range(x):
    # Input follows the existing Android/PyTorch contract:
    # rgb = (raw_0_1 - ImageNet_mean) / ImageNet_std.
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    raw_0_1 = x * std + mean
    return raw_0_1 * 2.0 - 1.0


def build_dual_mobilenetv2(rgb_weights="imagenet", dropout=0.2):
    # Prefix names keep the TFLite signature/input list ordered as RGB first, IR second.
    rgb_input = keras.Input(shape=(224, 224, 3), name="a_rgb")
    ir_input = keras.Input(shape=(224, 224, 1), name="b_ir")

    rgb_preprocessed = layers.Lambda(
        _rgb_current_norm_to_mobilenet_range,
        name="rgb_to_mobilenet_range",
    )(rgb_input)

    rgb_backbone = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=rgb_weights,
        pooling="avg",
        name="rgb_mobilenetv2",
    )
    ir_backbone = keras.applications.MobileNetV2(
        input_shape=(224, 224, 1),
        include_top=False,
        weights=None,
        pooling="avg",
        name="ir_mobilenetv2",
    )

    rgb_features = rgb_backbone(rgb_preprocessed)
    ir_features = ir_backbone(ir_input)
    fused = layers.Concatenate(name="fused_features")([rgb_features, ir_features])
    if dropout > 0:
        fused = layers.Dropout(dropout, name="classifier_dropout")(fused)
    logits = layers.Dense(len(CLASS_NAMES), name="logits")(fused)
    return keras.Model(inputs=[rgb_input, ir_input], outputs=logits, name="dual_mobilenetv2")


if __name__ == "__main__":
    model = build_dual_mobilenetv2(rgb_weights=None)
    model.summary()
    out = model(
        [
            tf.zeros((1, 224, 224, 3), dtype=tf.float32),
            tf.zeros((1, 224, 224, 1), dtype=tf.float32),
        ],
        training=False,
    )
    print("output shape:", out.shape)
