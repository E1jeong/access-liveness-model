"""Keras 안티스푸핑 모델 정의.

`--model-type`은 Android 입력 계약(dual/crop_rgb/crop_ir), `--backbone`은
특징 추출기(MobileNetV2/EfficientNet-Lite0/MobileFaceNet)를 뜻한다.
"""
import argparse
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classes import CLASS_NAMES
from keras_pipeline.efficientnet_lite import EfficientNetLite0
from keras_pipeline.mobilefacenet import MobileFaceNet
from keras_pipeline.spec import MODEL_INPUT_SIGNATURES, RGB_MEAN, RGB_STD


SUPPORTED_BACKBONES = ("mobilenetv2", "efficientnet_lite0", "mobilefacenet")
IMAGENET_BACKBONES = ("mobilenetv2", "efficientnet_lite0")


def _transfer_imagenet_weights_to_gray_backbone(source_backbone, gray_backbone, label, reduction="sum"):
    """3채널 ImageNet stem을 `sum`/`mean`으로 접어 1채널 백본에 이식한다."""
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
        if source_layer.name in ("Conv1", "stem_conv"):
            kernel = source_weights[0]
            reduced = kernel.mean(axis=2, keepdims=True) if reduction == "mean" else kernel.sum(axis=2, keepdims=True)
            target_layer.set_weights([reduced])
            copied += 1
            continue
        target_weights = target_layer.get_weights()
        if len(source_weights) == len(target_weights) and all(
            source.shape == target.shape for source, target in zip(source_weights, target_weights)
        ):
            target_layer.set_weights(source_weights)
            copied += 1
    print(f"[{label} backbone] copied ImageNet weights (reduction={reduction}) into {copied} layers")


def _make_backbone(backbone, input_shape, weights, pooling, name):
    if backbone == "mobilenetv2":
        return keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights=weights, pooling=pooling, name=name
        )
    if backbone == "efficientnet_lite0":
        return EfficientNetLite0(input_shape=input_shape, weights=weights, pooling=pooling, name=name)
    if backbone == "mobilefacenet":
        return MobileFaceNet(input_shape=input_shape, weights=weights, pooling=pooling, name=name)
    raise ValueError(f"Unknown backbone: {backbone}")


def _build_classifier_head(x, classifier_units, dropout, classifier_as_conv):
    if classifier_as_conv:
        if len(x.shape) == 2:
            x = layers.Reshape((1, 1, x.shape[-1]), name="fused_reshape_4d")(x)
        if classifier_units > 0:
            x = layers.Conv2D(classifier_units, 1, activation="relu", name="classifier_dense_conv")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name="classifier_dropout")(x)
        logits = layers.Conv2D(len(CLASS_NAMES), 1, name="logits_conv")(x)
        return layers.Reshape((len(CLASS_NAMES),), name="logits")(logits)
    if classifier_units > 0:
        x = layers.Dense(classifier_units, activation="relu", name="classifier_dense")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name="classifier_dropout")(x)
    return layers.Dense(len(CLASS_NAMES), name="logits")(x)


def _rgb_current_norm_to_mobilenet_range(x):
    raw_0_1 = x * tf.constant(RGB_STD, dtype=tf.float32) + tf.constant(RGB_MEAN, dtype=tf.float32)
    return raw_0_1 * 2.0 - 1.0


def _features_for_head(backbone_model, inputs, backbone, average_pool_op, prefix):
    features = backbone_model(inputs)
    if not average_pool_op:
        return features
    if backbone == "mobilefacenet":
        # GDConv이 이미 1x1 공간 위치를 만든다. export 경로에서는 4D를 보존한다.
        return features
    channels = 1280
    features = layers.AveragePooling2D(pool_size=(7, 7), name=f"{prefix}_average_pool")(features)
    return layers.Reshape((channels,), name=f"{prefix}_reshape")(features)


def build_dual_model(
    rgb_weights="imagenet", dropout=0.2, classifier_units=1024, gray_imagenet_init=True,
    rgb_input_mobilenet_range=False, average_pool_op=False, fixed_batch_size=None,
    classifier_as_conv=False, conv1_reduction="sum", backbone="mobilenetv2",
):
    if backbone == "mobilefacenet":
        raise ValueError("MobileFaceNet은 crop_ir 단일 입력만 지원합니다")
    rgb_name, rgb_shape = MODEL_INPUT_SIGNATURES["dual"][0]
    ir_name, ir_shape = MODEL_INPUT_SIGNATURES["dual"][1]
    rgb_input = keras.Input(batch_size=fixed_batch_size, shape=rgb_shape, name=rgb_name)
    ir_input = keras.Input(batch_size=fixed_batch_size, shape=ir_shape, name=ir_name)
    rgb_preprocessed = rgb_input if rgb_input_mobilenet_range else layers.Lambda(
        _rgb_current_norm_to_mobilenet_range, name="rgb_to_mobilenet_range"
    )(rgb_input)
    pooling = None if average_pool_op else "avg"
    rgb_backbone = _make_backbone(backbone, rgb_shape, rgb_weights, pooling, f"rgb_{backbone}")
    ir_backbone = _make_backbone(backbone, ir_shape, None, pooling, f"ir_{backbone}")
    if rgb_weights is not None and gray_imagenet_init:
        _transfer_imagenet_weights_to_gray_backbone(rgb_backbone, ir_backbone, "IR", conv1_reduction)
    rgb_features = _features_for_head(rgb_backbone, rgb_preprocessed, backbone, average_pool_op, "rgb")
    ir_features = _features_for_head(ir_backbone, ir_input, backbone, average_pool_op, "ir")
    logits = _build_classifier_head(layers.Concatenate(name="fused_features")([rgb_features, ir_features]), classifier_units, dropout, classifier_as_conv)
    return keras.Model([rgb_input, ir_input], logits, name=f"dual_{backbone}")


def build_single_model(
    input_type="crop_rgb", rgb_weights="imagenet", dropout=0.2, classifier_units=1024,
    gray_imagenet_init=True, rgb_input_mobilenet_range=False, average_pool_op=False,
    fixed_batch_size=None, classifier_as_conv=False, conv1_reduction="sum", backbone="mobilenetv2",
):
    if input_type not in ("crop_rgb", "crop_ir"):
        raise ValueError(f"Unknown input_type: {input_type}")
    if backbone == "mobilefacenet" and input_type != "crop_ir":
        raise ValueError("MobileFaceNet은 crop_ir 단일 입력만 지원합니다")
    if backbone == "mobilefacenet" and rgb_weights is not None:
        raise ValueError("MobileFaceNet은 scratch 학습만 지원하므로 --rgb-weights none을 사용해야 합니다")

    input_name, input_shape = MODEL_INPUT_SIGNATURES[input_type][0]
    model_input = keras.Input(batch_size=fixed_batch_size, shape=input_shape, name=input_name)
    pooling = None if average_pool_op else "avg"
    if input_type == "crop_rgb":
        backbone_input = model_input if rgb_input_mobilenet_range else layers.Lambda(
            _rgb_current_norm_to_mobilenet_range, name="rgb_to_mobilenet_range"
        )(model_input)
        backbone_model = _make_backbone(backbone, input_shape, rgb_weights, pooling, f"{input_type}_{backbone}")
    else:
        backbone_input = model_input
        backbone_model = _make_backbone(backbone, input_shape, None, pooling, f"{input_type}_{backbone}")
        if backbone in IMAGENET_BACKBONES and rgb_weights is not None and gray_imagenet_init:
            source = _make_backbone(backbone, (input_shape[0], input_shape[1], 3), rgb_weights, None, f"temp_rgb_{backbone}")
            _transfer_imagenet_weights_to_gray_backbone(source, backbone_model, "IR", conv1_reduction)
    features = _features_for_head(backbone_model, backbone_input, backbone, average_pool_op, input_type)
    logits = _build_classifier_head(features, classifier_units, dropout, classifier_as_conv)
    return keras.Model(model_input, logits, name=f"single_{input_type}_{backbone}")


# 기존 호출부 및 저장된 코드의 import 호환성.
build_dual_mobilenetv2 = build_dual_model
build_single_mobilenetv2 = build_single_model


def parse_args():
    parser = argparse.ArgumentParser(description="Keras 안티스푸핑 모델을 만들고 구조를 출력합니다.")
    parser.add_argument("--rgb-weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--classifier-units", type=int, default=1024)
    parser.add_argument("--no-gray-imagenet-init", action="store_true")
    parser.add_argument("--model-type", choices=["dual", "crop_rgb", "crop_ir"], default="dual")
    parser.add_argument("--backbone", choices=SUPPORTED_BACKBONES, default="mobilenetv2")
    parser.add_argument("--conv1-reduction", choices=["mean", "sum"], default="sum")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    weights = None if args.rgb_weights == "none" else args.rgb_weights
    if args.backbone == "mobilefacenet":
        weights = None
    builder = build_dual_model if args.model_type == "dual" else build_single_model
    kwargs = dict(rgb_weights=weights, dropout=args.dropout, classifier_units=args.classifier_units,
                  gray_imagenet_init=not args.no_gray_imagenet_init, conv1_reduction=args.conv1_reduction,
                  backbone=args.backbone)
    model = builder(**kwargs) if args.model_type == "dual" else builder(input_type=args.model_type, **kwargs)
    model.summary()
