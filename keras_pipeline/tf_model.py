import sys
import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classes import CLASS_NAMES


def _transfer_imagenet_weights_to_ir_backbone(source_backbone, ir_backbone):
    """Initialize a 1-channel MobileNetV2 from 3-channel ImageNet weights."""
    copied = 0
    for source_layer in source_backbone.layers:
        try:
            target_layer = ir_backbone.get_layer(source_layer.name)
        except ValueError:
            continue

        source_weights = source_layer.get_weights()
        if not source_weights:
            continue

        if source_layer.name == "Conv1":
            kernel = source_weights[0]
            averaged_kernel = kernel.mean(axis=2, keepdims=True)
            target_layer.set_weights([averaged_kernel])
            copied += 1
            continue

        target_weights = target_layer.get_weights()
        if len(source_weights) != len(target_weights):
            continue
        if all(sw.shape == tw.shape for sw, tw in zip(source_weights, target_weights)):
            target_layer.set_weights(source_weights)
            copied += 1

    print(f"[IR backbone] copied ImageNet weights into {copied} MobileNetV2 layers")


def _rgb_current_norm_to_mobilenet_range(x):
    # Input follows the existing Android/PyTorch contract:
    # rgb = (raw_0_1 - ImageNet_mean) / ImageNet_std.
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    raw_0_1 = x * std + mean
    return raw_0_1 * 2.0 - 1.0


def build_dual_mobilenetv2(
    rgb_weights="imagenet",
    dropout=0.2,
    classifier_units=1024,
    ir_imagenet_init=True,
    rgb_input_mobilenet_range=False,
    average_pool_op=False,
    fixed_batch_size=None,
):
    # Prefix names keep the TFLite signature/input list ordered as RGB first, IR second.
    rgb_input = keras.Input(batch_size=fixed_batch_size, shape=(224, 224, 3), name="a_rgb")
    ir_input = keras.Input(batch_size=fixed_batch_size, shape=(224, 224, 1), name="b_ir")

    if rgb_input_mobilenet_range:
        rgb_preprocessed = rgb_input
    else:
        rgb_preprocessed = layers.Lambda(
            _rgb_current_norm_to_mobilenet_range,
            name="rgb_to_mobilenet_range",
        )(rgb_input)

    rgb_backbone = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=rgb_weights,
        pooling=None if average_pool_op else "avg",
        name="rgb_mobilenetv2",
    )
    ir_backbone = keras.applications.MobileNetV2(
        input_shape=(224, 224, 1),
        include_top=False,
        weights=None,
        pooling=None if average_pool_op else "avg",
        name="ir_mobilenetv2",
    )
    if rgb_weights == "imagenet" and ir_imagenet_init:
        _transfer_imagenet_weights_to_ir_backbone(rgb_backbone, ir_backbone)

    rgb_features = rgb_backbone(rgb_preprocessed)
    ir_features = ir_backbone(ir_input)
    if average_pool_op:
        rgb_features = layers.AveragePooling2D(pool_size=(7, 7), name="rgb_average_pool")(rgb_features)
        rgb_features = layers.Reshape((1280,), name="rgb_reshape")(rgb_features)
        ir_features = layers.AveragePooling2D(pool_size=(7, 7), name="ir_average_pool")(ir_features)
        ir_features = layers.Reshape((1280,), name="ir_reshape")(ir_features)
    fused = layers.Concatenate(name="fused_features")([rgb_features, ir_features])
    if classifier_units > 0:
        fused = layers.Dense(classifier_units, activation="relu", name="classifier_dense")(fused)
    if dropout > 0:
        fused = layers.Dropout(dropout, name="classifier_dropout")(fused)
    logits = layers.Dense(len(CLASS_NAMES), name="logits")(fused)
    return keras.Model(inputs=[rgb_input, ir_input], outputs=logits, name="dual_mobilenetv2")


def parse_args():
    parser = argparse.ArgumentParser(description="Build and summarize the Keras dual MobileNetV2 model.")
    parser.add_argument("--rgb-weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--classifier-units", type=int, default=1024)
    parser.add_argument("--no-ir-imagenet-init", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rgb_weights = None if args.rgb_weights == "none" else args.rgb_weights
    model = build_dual_mobilenetv2(
        rgb_weights=rgb_weights,
        dropout=args.dropout,
        classifier_units=args.classifier_units,
        ir_imagenet_init=not args.no_ir_imagenet_init,
    )
    model.summary()
    out = model(
        [
            tf.zeros((1, 224, 224, 3), dtype=tf.float32),
            tf.zeros((1, 224, 224, 1), dtype=tf.float32),
        ],
        training=False,
    )
    print("output shape:", out.shape)
