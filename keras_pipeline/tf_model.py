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
from keras_pipeline.spec import MODEL_INPUT_SIGNATURES, RGB_MEAN, RGB_STD


def _transfer_imagenet_weights_to_gray_backbone(source_backbone, gray_backbone, label, reduction="mean"):
    """Initialize a 1-channel MobileNetV2 from a 3-channel MobileNetV2."""
    if reduction not in ("mean", "sum"):
        raise ValueError(f"Unknown reduction: {reduction}")
    copied = 0
    for source_layer in source_backbone.layers:
        try:
            target_layer = gray_backbone.get_layer(source_layer.name)
        except ValueError:
            continue

        source_weights = source_layer.get_weights()
        if not source_weights:
            continue

        if source_layer.name == "Conv1":
            kernel = source_weights[0]
            if reduction == "mean":
                transferred_kernel = kernel.mean(axis=2, keepdims=True)
            else:  # sum
                transferred_kernel = kernel.sum(axis=2, keepdims=True)
            target_layer.set_weights([transferred_kernel])
            copied += 1
            continue

        target_weights = target_layer.get_weights()
        if len(source_weights) != len(target_weights):
            continue
        if all(sw.shape == tw.shape for sw, tw in zip(source_weights, target_weights)):
            target_layer.set_weights(source_weights)
            copied += 1

    print(f"[{label} backbone] copied ImageNet weights (reduction={reduction}) into {copied} MobileNetV2 layers")


# Unused face weight loader helper functions were removed during hardening refactor.


def _build_classifier_head(x, classifier_units, dropout, num_classes, classifier_as_conv, dtype=None):
    if classifier_as_conv:
        if len(x.shape) == 2:
            x = layers.Reshape((1, 1, x.shape[-1]), name="fused_reshape_4d")(x)
        if classifier_units > 0:
            x = layers.Conv2D(classifier_units, kernel_size=(1, 1), activation="relu", name="classifier_dense_conv")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name="classifier_dropout")(x)
        logits_4d = layers.Conv2D(num_classes, kernel_size=(1, 1), name="logits_conv", dtype=dtype)(x)
        return layers.Reshape((num_classes,), name="logits", dtype=dtype)(logits_4d)
    else:
        if classifier_units > 0:
            x = layers.Dense(classifier_units, activation="relu", name="classifier_dense")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name="classifier_dropout")(x)
        return layers.Dense(num_classes, name="logits", dtype=dtype)(x)


def _rgb_current_norm_to_mobilenet_range(x):
    # Input follows the existing Android/PyTorch contract:
    # rgb = (raw_0_1 - ImageNet_mean) / ImageNet_std.
    mean = tf.constant(RGB_MEAN, dtype=tf.float32)
    std = tf.constant(RGB_STD, dtype=tf.float32)
    raw_0_1 = x * std + mean
    return raw_0_1 * 2.0 - 1.0


def build_dual_mobilenetv2(
    rgb_weights="imagenet",
    dropout=0.2,
    classifier_units=1024,
    gray_imagenet_init=True,
    rgb_input_mobilenet_range=False,
    average_pool_op=False,
    fixed_batch_size=None,
    classifier_as_conv=False,
    conv1_reduction="mean",
):
    # Prefix names keep the TFLite signature/input list ordered as RGB first, IR second.
    rgb_name, rgb_shape = MODEL_INPUT_SIGNATURES["dual"][0]
    ir_name, ir_shape = MODEL_INPUT_SIGNATURES["dual"][1]
    rgb_input = keras.Input(batch_size=fixed_batch_size, shape=rgb_shape, name=rgb_name)
    ir_input = keras.Input(batch_size=fixed_batch_size, shape=ir_shape, name=ir_name)

    if rgb_input_mobilenet_range:
        rgb_preprocessed = rgb_input
    else:
        rgb_preprocessed = layers.Lambda(
            _rgb_current_norm_to_mobilenet_range,
            name="rgb_to_mobilenet_range",
        )(rgb_input)

    rgb_backbone = keras.applications.MobileNetV2(
        input_shape=rgb_shape,
        include_top=False,
        weights=rgb_weights,
        pooling=None if average_pool_op else "avg",
        name="rgb_mobilenetv2",
    )
    ir_backbone = keras.applications.MobileNetV2(
        input_shape=ir_shape,
        include_top=False,
        weights=None,
        pooling=None if average_pool_op else "avg",
        name="ir_mobilenetv2",
    )

    if rgb_weights is not None and gray_imagenet_init:
        _transfer_imagenet_weights_to_gray_backbone(rgb_backbone, ir_backbone, "IR", reduction=conv1_reduction)

    rgb_features = rgb_backbone(rgb_preprocessed)
    ir_features = ir_backbone(ir_input)
    if average_pool_op:
        rgb_features = layers.AveragePooling2D(pool_size=(7, 7), name="rgb_average_pool")(rgb_features)
        rgb_features = layers.Reshape((1280,), name="rgb_reshape")(rgb_features)
        ir_features = layers.AveragePooling2D(pool_size=(7, 7), name="ir_average_pool")(ir_features)
        ir_features = layers.Reshape((1280,), name="ir_reshape")(ir_features)
    fused = layers.Concatenate(name="fused_features")([rgb_features, ir_features])
    logits = _build_classifier_head(fused, classifier_units, dropout, len(CLASS_NAMES), classifier_as_conv)
    return keras.Model(inputs=[rgb_input, ir_input], outputs=logits, name="dual_mobilenetv2")


def build_single_mobilenetv2(
    input_type="crop_rgb",
    rgb_weights="imagenet",
    dropout=0.2,
    classifier_units=1024,
    gray_imagenet_init=True,
    rgb_input_mobilenet_range=False,
    average_pool_op=False,
    fixed_batch_size=None,
    classifier_as_conv=False,
    conv1_reduction="mean",
):
    if input_type not in ("crop_rgb", "crop_ir"):
        raise ValueError(f"Unknown input_type: {input_type}")

    input_name, input_shape = MODEL_INPUT_SIGNATURES[input_type][0]

    if input_type == "crop_rgb":
        rgb_input = keras.Input(batch_size=fixed_batch_size, shape=input_shape, name=input_name)
        if rgb_input_mobilenet_range:
            rgb_preprocessed = rgb_input
        else:
            rgb_preprocessed = layers.Lambda(
                _rgb_current_norm_to_mobilenet_range,
                name="rgb_to_mobilenet_range",
            )(rgb_input)

        rgb_backbone = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=rgb_weights,
            pooling=None if average_pool_op else "avg",
            name="crop_rgb_mobilenetv2",
        )

        features = rgb_backbone(rgb_preprocessed)
        if average_pool_op:
            features = layers.AveragePooling2D(pool_size=(7, 7), name="crop_rgb_average_pool")(features)
            features = layers.Reshape((1280,), name="crop_rgb_reshape")(features)

        inputs = rgb_input

    else: # crop_ir
        ir_input = keras.Input(batch_size=fixed_batch_size, shape=input_shape, name=input_name)
        ir_backbone = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=None,
            pooling=None if average_pool_op else "avg",
            name="crop_ir_mobilenetv2",
        )

        if rgb_weights is not None and gray_imagenet_init:
            temp_rgb_backbone = keras.applications.MobileNetV2(
                input_shape=(input_shape[0], input_shape[1], 3),
                include_top=False,
                weights=rgb_weights,
                pooling=None,
                name="temp_rgb_mobilenetv2",
            )
            _transfer_imagenet_weights_to_gray_backbone(temp_rgb_backbone, ir_backbone, "IR", reduction=conv1_reduction)

        features = ir_backbone(ir_input)
        if average_pool_op:
            features = layers.AveragePooling2D(pool_size=(7, 7), name="crop_ir_average_pool")(features)
            features = layers.Reshape((1280,), name="crop_ir_reshape")(features)

        inputs = ir_input

    logits = _build_classifier_head(features, classifier_units, dropout, len(CLASS_NAMES), classifier_as_conv)

    return keras.Model(inputs=inputs, outputs=logits, name=f"single_{input_type}_mobilenetv2")


def _make_backbone(input_shape, weights, pooling, name):
    return keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
        pooling=pooling,
        name=name,
    )


def _pool_backbone_output(features, prefix):
    features = layers.AveragePooling2D(pool_size=(7, 7), name=f"{prefix}_average_pool")(features)
    return layers.Reshape((1280,), name=f"{prefix}_reshape")(features)





def parse_args():
    parser = argparse.ArgumentParser(description="Build and summarize the Keras MobileNetV2 models.")
    parser.add_argument("--rgb-weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--classifier-units", type=int, default=1024)
    parser.add_argument("--no-gray-imagenet-init", action="store_true")
    parser.add_argument(
        "--model-type",
        choices=["dual", "crop_rgb", "crop_ir"],
        default="dual",
        help="학습할 모델 종류 (dual: 2입력, crop_rgb: 단일 RGB, crop_ir: 단일 IR)"
    )
    parser.add_argument(
        "--conv1-reduction",
        choices=["mean", "sum"],
        default="mean",
        help="1채널 Conv1 가중치 이식 시 축소 방식 (mean: 평균, sum: 합산)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rgb_weights = None if args.rgb_weights == "none" else args.rgb_weights
    if args.model_type == "dual":
        model = build_dual_mobilenetv2(
            rgb_weights=rgb_weights,
            dropout=args.dropout,
            classifier_units=args.classifier_units,
            gray_imagenet_init=not args.no_gray_imagenet_init,
            conv1_reduction=args.conv1_reduction,
        )
        dummy_inputs = [
            tf.zeros((1, 224, 224, 3), dtype=tf.float32),
            tf.zeros((1, 224, 224, 1), dtype=tf.float32),
        ]
    else:
        model = build_single_mobilenetv2(
            input_type=args.model_type,
            rgb_weights=rgb_weights,
            dropout=args.dropout,
            classifier_units=args.classifier_units,
            gray_imagenet_init=not args.no_gray_imagenet_init,
            conv1_reduction=args.conv1_reduction,
        )
        if args.model_type == "crop_rgb":
            dummy_inputs = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
        else:
            dummy_inputs = tf.zeros((1, 224, 224, 1), dtype=tf.float32)
    model.summary()
    out = model(dummy_inputs, training=False)
    print("output shape:", out.shape)
