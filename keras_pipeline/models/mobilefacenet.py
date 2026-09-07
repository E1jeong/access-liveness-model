"""224×224 IR anti-spoofing용 ReLU6 MobileFaceNet 백본.

원 MobileFaceNet은 112 입력에서 7×7 GDConv을 사용한다. 이 프로젝트의 고정
224 입력 계약에서는 첫 depthwise layer의 stride를 2로 해 GDConv 직전 7×7을 유지한다.
외부 얼굴 인식 가중치는 사용하지 않고 항상 scratch로 초기화한다.
"""
from typing import Optional, Tuple

from tensorflow import keras
from tensorflow.keras import layers


BLOCK_ARGS = [
    # (expansion, output channels, repeats, first stride)
    (2, 64, 5, 2),
    (4, 128, 1, 2),
    (2, 128, 6, 1),
    (4, 128, 1, 2),
    (2, 128, 2, 1),
]


def _conv_bn_relu6(x, filters, kernel_size, strides, name, groups=1, linear=False, padding="same"):
    if groups == 1:
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, name=name)(x)
    else:
        x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding=padding, use_bias=False, name=name)(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    if not linear:
        x = layers.ReLU(max_value=6.0, name=f"{name}_relu6")(x)
    return x


def _bottleneck(inputs, expansion, out_channels, stride, name):
    in_channels = int(inputs.shape[-1])
    x = inputs
    if expansion != 1:
        x = _conv_bn_relu6(x, in_channels * expansion, 1, 1, f"{name}_expand")
    x = _conv_bn_relu6(x, int(x.shape[-1]), 3, stride, f"{name}_depthwise", groups=int(x.shape[-1]))
    x = _conv_bn_relu6(x, out_channels, 1, 1, f"{name}_project", linear=True)
    if stride == 1 and in_channels == out_channels:
        x = layers.Add(name=f"{name}_add")([inputs, x])
    return x


def MobileFaceNet(
    input_shape: Tuple[int, int, int] = (224, 224, 1),
    include_top: bool = False,
    weights: Optional[str] = None,
    input_tensor=None,
    pooling: Optional[str] = None,
    name: str = "mobilefacenet",
):
    """분류 헤드 앞의 MobileFaceNet 특징 추출기를 만든다."""
    if include_top:
        raise ValueError("MobileFaceNet backbone은 include_top=False만 지원합니다")
    if weights is not None:
        raise ValueError("MobileFaceNet은 외부 pretrained weights를 지원하지 않습니다")
    if input_shape[-1] != 1:
        raise ValueError("MobileFaceNet은 crop_ir 1채널 입력만 지원합니다")
    if input_shape[:2] != (224, 224):
        raise ValueError("MobileFaceNet은 고정 224x224 입력만 지원합니다")

    inputs = input_tensor if input_tensor is not None else layers.Input(shape=input_shape, name=f"{name}_input")
    x = _conv_bn_relu6(inputs, 64, 3, 2, "conv1")
    # 224 입력을 원 MobileFaceNet의 GDConv 직전 7x7까지 내리기 위한 유일한 적응.
    x = _conv_bn_relu6(x, 64, 3, 2, "conv2_depthwise", groups=64)
    for stage, (expansion, channels, repeats, stride) in enumerate(BLOCK_ARGS, start=1):
        for repeat in range(repeats):
            x = _bottleneck(x, expansion, channels, stride if repeat == 0 else 1, f"bottleneck_{stage}_{repeat}")
    x = _conv_bn_relu6(x, 512, 1, 1, "conv3")
    if tuple(x.shape[1:3]) != (7, 7):
        raise ValueError(f"MobileFaceNet GDConv 입력은 7x7이어야 합니다: {x.shape}")
    # GDConv은 위치별 channel weight를 유지하는 depthwise 7x7 선형 연산이다.
    x = _conv_bn_relu6(x, 512, 7, 1, "gdconv", groups=512, linear=True, padding="valid")
    x = _conv_bn_relu6(x, 128, 1, 1, "embedding", linear=True)
    if pooling == "avg":
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    elif pooling == "max":
        x = layers.GlobalMaxPooling2D(name="max_pool")(x)
    elif pooling is not None:
        raise ValueError(f"Unknown pooling: {pooling}")
    return keras.Model(inputs, x, name=name)
