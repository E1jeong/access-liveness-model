"""Keras 안티스푸핑 모델 정의.

`--model-type`은 Android 입력 계약(dual/crop_rgb/crop_ir), `--backbone`은
특징 추출기(MobileNetV2/EfficientNet-Lite0)를 뜻한다.

Multi-Task Auxiliary 3D Depth 학습 지원:
  - `aux_depth=True` 시 12-Class `logits` 외에 14x14 `depth_output` 헤드가 함께 생성된다.
  - `aux_residual=True` 시 IR 고주파 에너지를 예측하는 14x14 `residual_output` 헤드가 함께 생성된다.
  - `aux_binary_pad=True` 시 Phase 2 bona-fide/spoof용 `pad_output` head가 함께 생성된다.
  - `aux_supcon=True` 시 학습 전용 L2 정규화 projection head가 함께 생성된다.
  - `extract_deploy_model(model)`을 호출하면 보조 헤드를 제거한 순수 배포용 단일 출력 모델을 얻을 수 있다.
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

from common.classes import CLASS_NAMES
from keras_pipeline.models.efficientnet_lite import EfficientNetLite0
from keras_pipeline.data.spec import MODEL_INPUT_SIGNATURES, RGB_MEAN, RGB_STD


SUPPORTED_BACKBONES = ("mobilenetv2", "efficientnet_lite0")
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


def _make_backbone(backbone, input_shape, weights, pooling, name=None):
    if backbone == "mobilenetv2":
        base = keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights=weights, pooling=pooling
        )
        if name:
            base = keras.Model(inputs=base.input, outputs=base.output, name=name)
        return base
    if backbone == "efficientnet_lite0":
        return EfficientNetLite0(input_shape=input_shape, weights=weights, pooling=pooling, name=name)
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


def _build_depth_head(spatial_features, prefix="aux"):
    """중간 공간 특징 맵(7x7)으로부터 14x14x1 3D 깊이 지도를 예측하는 경량 보조 헤드."""
    # 7x7 -> 14x14 업샘플링 및 합성곱
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"{prefix}_depth_upsample")(spatial_features)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name=f"{prefix}_depth_conv1")(x)
    # 0.0 ~ 1.0 깊이 값 예측을 위한 Sigmoid
    depth_out = layers.Conv2D(1, 3, padding="same", activation="sigmoid", name="depth_output")(x)
    return depth_out


def _build_residual_head(spatial_features, prefix="aux"):
    """Predict a 14x14 normalized high-frequency energy map from 7x7 features."""
    x = layers.UpSampling2D(
        size=(2, 2), interpolation="bilinear", name=f"{prefix}_residual_upsample"
    )(spatial_features)
    x = layers.Conv2D(
        64, 3, padding="same", activation="relu", name=f"{prefix}_residual_conv1"
    )(x)
    return layers.Conv2D(
        1, 3, padding="same", activation="sigmoid", name="residual_output"
    )(x)


def _build_binary_pad_head(features):
    """학습 전용 Phase 2 bona-fide/spoof logits head."""
    return layers.Dense(1, name="pad_output")(features)


def _build_supcon_head(features, projection_dim):
    """학습 전용 SupCon projection head."""
    x = layers.Dense(projection_dim, activation="relu", name="supcon_projection_hidden")(features)
    x = layers.Dense(projection_dim, use_bias=False, name="supcon_projection")(x)
    return layers.UnitNormalization(axis=-1, name="supcon_output")(x)


def _build_training_outputs(
    logits, depth_out=None, residual_out=None, pad_out=None, supcon_out=None
):
    outputs = [logits]
    if depth_out is not None:
        outputs.append(depth_out)
    if residual_out is not None:
        outputs.append(residual_out)
    if pad_out is not None:
        outputs.append(pad_out)
    if supcon_out is not None:
        outputs.append(supcon_out)
    return outputs[0] if len(outputs) == 1 else outputs


def _rgb_current_norm_to_mobilenet_range(x):
    raw_0_1 = x * tf.constant(RGB_STD, dtype=tf.float32) + tf.constant(RGB_MEAN, dtype=tf.float32)
    return raw_0_1 * 2.0 - 1.0


def _features_for_head(backbone_model, inputs, average_pool_op, prefix):
    features = backbone_model(inputs)
    if not average_pool_op:
        return features
    channels = 1280
    features = layers.AveragePooling2D(pool_size=(7, 7), name=f"{prefix}_average_pool")(features)
    return layers.Reshape((channels,), name=f"{prefix}_reshape")(features)


def extract_deploy_model(model):
    """보조 헤드(depth_output 등)를 제거하고 순수 12-Class logits만 출력하는 배포용 모델을 반환한다."""
    try:
        logits_tensor = model.get_layer("logits").output
    except ValueError:
        # logits 레이어가 없으면 모델 전체가 이미 단일 출력이므로 그대로 반환
        return model
    return keras.Model(inputs=model.inputs, outputs=logits_tensor, name=f"{model.name}_deploy")


def build_dual_model(
    rgb_weights="imagenet", dropout=0.2, classifier_units=1024, gray_imagenet_init=True,
    rgb_input_mobilenet_range=False, average_pool_op=False, fixed_batch_size=None,
    classifier_as_conv=False, conv1_reduction="sum", backbone="mobilenetv2",
    aux_depth=False, aux_binary_pad=False, aux_supcon=False, projection_dim=128, aux_residual=False,
):
    rgb_name, rgb_shape = MODEL_INPUT_SIGNATURES["dual"][0]
    ir_name, ir_shape = MODEL_INPUT_SIGNATURES["dual"][1]
    rgb_input = keras.Input(batch_size=fixed_batch_size, shape=rgb_shape, name=rgb_name)
    ir_input = keras.Input(batch_size=fixed_batch_size, shape=ir_shape, name=ir_name)
    rgb_preprocessed = rgb_input if rgb_input_mobilenet_range else layers.Lambda(
        _rgb_current_norm_to_mobilenet_range, name="rgb_to_mobilenet_range"
    )(rgb_input)
    
    pooling = None if (average_pool_op or aux_depth or aux_residual) else "avg"
    rgb_backbone = _make_backbone(backbone, rgb_shape, rgb_weights, pooling, f"rgb_{backbone}")
    ir_backbone = _make_backbone(backbone, ir_shape, None, pooling, f"ir_{backbone}")
    if rgb_weights is not None and gray_imagenet_init:
        _transfer_imagenet_weights_to_gray_backbone(rgb_backbone, ir_backbone, "IR", conv1_reduction)
    
    if aux_depth or aux_residual:
        rgb_raw = rgb_backbone(rgb_preprocessed)
        ir_raw = ir_backbone(ir_input)
        rgb_features = layers.GlobalAveragePooling2D(name="rgb_gap")(rgb_raw)
        ir_features = layers.GlobalAveragePooling2D(name="ir_gap")(ir_raw)
        fused_features = layers.Concatenate(name="fused_features")([rgb_features, ir_features])
        logits = _build_classifier_head(fused_features, classifier_units, dropout, classifier_as_conv)
        depth_out = _build_depth_head(ir_raw, prefix="dual_ir") if aux_depth else None
        residual_out = _build_residual_head(ir_raw, prefix="dual_ir") if aux_residual else None
        pad_out = _build_binary_pad_head(fused_features) if aux_binary_pad else None
        supcon_out = _build_supcon_head(fused_features, projection_dim) if aux_supcon else None
        return keras.Model(
            [rgb_input, ir_input],
            _build_training_outputs(logits, depth_out, residual_out, pad_out, supcon_out),
            name=f"dual_{backbone}",
        )
    else:
        rgb_features = _features_for_head(rgb_backbone, rgb_preprocessed, average_pool_op, "rgb")
        ir_features = _features_for_head(ir_backbone, ir_input, average_pool_op, "ir")
        fused_features = layers.Concatenate(name="fused_features")([rgb_features, ir_features])
        logits = _build_classifier_head(fused_features, classifier_units, dropout, classifier_as_conv)
        pad_out = _build_binary_pad_head(fused_features) if aux_binary_pad else None
        supcon_out = _build_supcon_head(fused_features, projection_dim) if aux_supcon else None
        return keras.Model(
            [rgb_input, ir_input], _build_training_outputs(logits, pad_out=pad_out, supcon_out=supcon_out),
            name=f"dual_{backbone}",
        )


def build_single_model(
    input_type="crop_rgb", rgb_weights="imagenet", dropout=0.2, classifier_units=1024,
    gray_imagenet_init=True, rgb_input_mobilenet_range=False, average_pool_op=False,
    fixed_batch_size=None, classifier_as_conv=False, conv1_reduction="sum", backbone="mobilenetv2",
    aux_depth=False, aux_binary_pad=False, aux_supcon=False, projection_dim=128, aux_residual=False,
):
    if input_type not in ("crop_rgb", "crop_ir"):
        raise ValueError(f"Unknown input_type: {input_type}")
    if aux_residual and input_type != "crop_ir":
        raise ValueError("High-frequency residual supervision is supported only for IR inputs.")

    input_name, input_shape = MODEL_INPUT_SIGNATURES[input_type][0]
    model_input = keras.Input(batch_size=fixed_batch_size, shape=input_shape, name=input_name)
    
    pooling = None if (average_pool_op or aux_depth or aux_residual) else "avg"
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

    if aux_depth or aux_residual:
        raw_features = backbone_model(backbone_input)
        features = layers.GlobalAveragePooling2D(name=f"{input_type}_gap")(raw_features)
        logits = _build_classifier_head(features, classifier_units, dropout, classifier_as_conv)
        depth_out = _build_depth_head(raw_features, prefix=input_type) if aux_depth else None
        residual_out = _build_residual_head(raw_features, prefix=input_type) if aux_residual else None
        pad_out = _build_binary_pad_head(features) if aux_binary_pad else None
        supcon_out = _build_supcon_head(features, projection_dim) if aux_supcon else None
        return keras.Model(
            model_input,
            _build_training_outputs(logits, depth_out, residual_out, pad_out, supcon_out),
            name=f"single_{input_type}_{backbone}",
        )
    else:
        features = _features_for_head(backbone_model, backbone_input, average_pool_op, input_type)
        logits = _build_classifier_head(features, classifier_units, dropout, classifier_as_conv)
        pad_out = _build_binary_pad_head(features) if aux_binary_pad else None
        supcon_out = _build_supcon_head(features, projection_dim) if aux_supcon else None
        return keras.Model(
            model_input, _build_training_outputs(logits, pad_out=pad_out, supcon_out=supcon_out),
            name=f"single_{input_type}_{backbone}",
        )


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
    parser.add_argument("--aux-depth", action="store_true", help="3D Depth 보조 헤드 생성 여부")
    parser.add_argument("--aux-residual", action="store_true", help="IR 고주파 잔차 지도 보조 헤드 생성 여부")
    parser.add_argument("--aux-binary-pad", action="store_true", help="Phase 2 binary PAD 보조 헤드 생성 여부")
    parser.add_argument("--aux-supcon", action="store_true", help="SupCon projection head 생성 여부")
    parser.add_argument("--projection-dim", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    weights = None if args.rgb_weights == "none" else args.rgb_weights
    builder = build_dual_model if args.model_type == "dual" else build_single_model
    kwargs = dict(rgb_weights=weights, dropout=args.dropout, classifier_units=args.classifier_units,
                  gray_imagenet_init=not args.no_gray_imagenet_init, conv1_reduction=args.conv1_reduction,
                  backbone=args.backbone, aux_depth=args.aux_depth, aux_binary_pad=args.aux_binary_pad,
                  aux_supcon=args.aux_supcon, aux_residual=args.aux_residual,
                  projection_dim=args.projection_dim)
    model = builder(**kwargs) if args.model_type == "dual" else builder(input_type=args.model_type, **kwargs)
    model.summary()
