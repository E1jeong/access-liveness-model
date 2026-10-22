"""NPU INT8 변환을 고려한 EfficientNet-Lite0 백본."""
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


WEIGHTS_URL_NOTOP = (
    "https://github.com/sebastian-sz/efficientnet-lite-keras/releases/download/v1.0/efficientnet_lite_b0_notop.h5"
)
WEIGHTS_HASH_NOTOP = "d7a91a3c0e7f0bdffe67f599ebe511cd"

# (kernel, input channels, output channels, expansion, stride, repeats, stage)
BLOCK_ARGS = [
    (3, 32, 16, 1, 1, 1, "1"),
    (3, 16, 24, 6, 2, 2, "2"),
    (5, 24, 40, 6, 2, 2, "3"),
    (3, 40, 80, 6, 2, 3, "4"),
    (5, 80, 112, 6, 1, 3, "5"),
    (5, 112, 192, 6, 2, 4, "6"),
    (3, 192, 320, 6, 1, 1, "7"),
]


def _mb_conv_block(inputs, in_channels, out_channels, kernel_size, expand_ratio, stride, prefix):
    x = inputs
    expanded_channels = in_channels * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(expanded_channels, 1, padding="same", use_bias=False, name=f"{prefix}_expand_conv")(x)
        x = layers.BatchNormalization(axis=-1, name=f"{prefix}_expand_bn")(x)
        x = layers.ReLU(max_value=6.0, name=f"{prefix}_expand_activation")(x)

    if stride == 2:
        padding = ((1, 1), (1, 1)) if kernel_size == 3 else ((2, 2), (2, 2))
        x = layers.ZeroPadding2D(padding=padding, name=f"{prefix}_dwconv_pad")(x)
        padding_mode = "valid"
    else:
        padding_mode = "same"
    x = layers.DepthwiseConv2D(kernel_size, strides=stride, padding=padding_mode, use_bias=False, name=f"{prefix}_dwconv")(x)
    x = layers.BatchNormalization(axis=-1, name=f"{prefix}_bn")(x)
    x = layers.ReLU(max_value=6.0, name=f"{prefix}_activation")(x)
    x = layers.Conv2D(out_channels, 1, padding="same", use_bias=False, name=f"{prefix}_project_conv")(x)
    x = layers.BatchNormalization(axis=-1, name=f"{prefix}_project_bn")(x)
    if stride == 1 and in_channels == out_channels:
        x = layers.Add(name=f"{prefix}_add")([inputs, x])
    return x


def EfficientNetLite0(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    include_top: bool = False,
    weights: Optional[str] = "imagenet",
    input_tensor=None,
    pooling: Optional[str] = None,
    name: str = "efficientnet_lite0",
):
    """ReLU6·SE 제거형 EfficientNet-Lite0 백본을 만든다."""
    if include_top:
        raise ValueError("EfficientNetLite0 backbone은 include_top=False만 지원합니다")
    if weights not in ("imagenet", None):
        raise ValueError("weights는 'imagenet' 또는 None이어야 합니다")

    img_input = input_tensor if input_tensor is not None else layers.Input(shape=input_shape, name=f"{name}_input")
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name="stem_conv_pad")(img_input)
    x = layers.Conv2D(32, 3, strides=2, padding="valid", use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(axis=-1, name="stem_bn")(x)
    x = layers.ReLU(max_value=6.0, name="stem_activation")(x)

    letters = "abcdefghijklmnopqrstuvwxyz"
    for kernel, in_channels, out_channels, expansion, stride, repeats, stage in BLOCK_ARGS:
        for repeat in range(repeats):
            x = _mb_conv_block(
                x,
                in_channels if repeat == 0 else out_channels,
                out_channels,
                kernel,
                expansion,
                stride if repeat == 0 else 1,
                f"block{stage}{letters[repeat]}",
            )

    x = layers.Conv2D(1280, 1, padding="same", use_bias=False, name="top_conv")(x)
    x = layers.BatchNormalization(axis=-1, name="top_bn")(x)
    x = layers.ReLU(max_value=6.0, name="top_activation")(x)
    if pooling == "avg":
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    elif pooling == "max":
        x = layers.GlobalMaxPooling2D(name="max_pool")(x)
    elif pooling is not None:
        raise ValueError(f"Unknown pooling: {pooling}")

    model = keras.Model(img_input, x, name=name)
    if weights == "imagenet" and input_shape[-1] == 3:
        weights_path = keras.utils.get_file(
            "efficientnet_lite_b0_notop.h5", WEIGHTS_URL_NOTOP, file_hash=WEIGHTS_HASH_NOTOP, cache_subdir="models"
        )
        model.load_weights(weights_path, by_name=True)
    elif weights == "imagenet":
        print(f"[{name}] 1채널 입력에는 ImageNet 가중치를 직접 로드하지 않습니다")
    return model
