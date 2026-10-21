"""YuNet-N model with optional five-point landmark supervision.

Derived from Shiqi Yu's ``libfacedetection.train`` at commit
``dca340aa082c71081a68d17db8e58b33a58a914b``. The upstream project is
licensed under BSD-3-Clause; see ``third_party/yunet/LICENSE``.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class YuNetOutputs(NamedTuple):
    """Raw stride-8/16/32 predictions, before decoding and NMS."""

    cls_scores: list[torch.Tensor]
    objectnesses: list[torch.Tensor]
    bbox_regressions: list[torch.Tensor]


class YuNetLandmarkOutputs(NamedTuple):
    """Box predictions plus five stride-relative (x, y) offsets per candidate."""

    cls_scores: list[torch.Tensor]
    objectnesses: list[torch.Tensor]
    bbox_regressions: list[torch.Tensor]
    landmark_regressions: list[torch.Tensor]


def _init_yunet_weights(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            if layer.bias is not None:
                nn.init.xavier_normal_(layer.weight.data)
                layer.bias.data.fill_(0.02)
            else:
                layer.weight.data.normal_(0, 0.01)
        elif isinstance(layer, nn.BatchNorm2d):
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()


class _ConvDPUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activate: bool = True):
        super().__init__()
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        self.depthwise = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1,
            groups=out_channels,
            bias=True,
        )
        self.activate = activate
        if activate:
            self.batch_norm = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.depthwise(self.pointwise(inputs))
        if self.activate:
            outputs = self.relu(self.batch_norm(outputs))
        return outputs


class _Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=True)
        self.batch_norm = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.depthwise_pointwise = _ConvDPUnit(16, 16)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.relu(self.batch_norm(self.conv(inputs)))
        return self.depthwise_pointwise(outputs)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            _ConvDPUnit(in_channels, in_channels),
            _ConvDPUnit(in_channels, out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class _YuNetNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                _Stem(),
                _ConvBlock(16, 64),
                _ConvBlock(64, 64),
                _ConvBlock(64, 64),
                _ConvBlock(64, 64),
                _ConvBlock(64, 64),
            ]
        )
        _init_yunet_weights(self)

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        features = inputs
        for index, stage in enumerate(self.stages):
            features = stage(features)
            if index in (3, 4, 5):
                outputs.append(features)
            if index in (0, 2, 3, 4):
                features = F.max_pool2d(features, 2)
        return outputs


class _TFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral_convs = nn.ModuleList([_ConvDPUnit(64, 64) for _ in range(3)])
        _init_yunet_weights(self)

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        features = list(inputs)
        for index in range(len(features) - 1, 0, -1):
            features[index] = self.lateral_convs[index](features[index])
            features[index - 1] = features[index - 1] + F.interpolate(
                features[index],
                size=features[index - 1].shape[-2:],
                mode="nearest",
            )
        features[0] = self.lateral_convs[0](features[0])
        return features


class _BoxOnlyHead(nn.Module):
    def __init__(self, landmarks: bool = False):
        super().__init__()
        self.shared_convs = nn.ModuleList([_ConvDPUnit(64, 64) for _ in range(3)])
        self.cls_heads = nn.ModuleList(
            [_ConvDPUnit(64, 1, activate=False) for _ in range(3)]
        )
        self.objectness_heads = nn.ModuleList(
            [_ConvDPUnit(64, 1, activate=False) for _ in range(3)]
        )
        self.bbox_heads = nn.ModuleList(
            [_ConvDPUnit(64, 4, activate=False) for _ in range(3)]
        )
        if landmarks:
            self.landmark_heads = nn.ModuleList(
                [_ConvDPUnit(64, 10, activate=False) for _ in range(3)]
            )
        _init_yunet_weights(self)

    def forward(self, inputs: list[torch.Tensor]) -> YuNetOutputs | YuNetLandmarkOutputs:
        shared = [layer(features) for layer, features in zip(self.shared_convs, inputs)]
        outputs = YuNetOutputs(
            cls_scores=[layer(features) for layer, features in zip(self.cls_heads, shared)],
            objectnesses=[
                layer(features) for layer, features in zip(self.objectness_heads, shared)
            ],
            bbox_regressions=[
                layer(features) for layer, features in zip(self.bbox_heads, shared)
            ],
        )
        if hasattr(self, "landmark_heads"):
            return YuNetLandmarkOutputs(
                *outputs,
                [layer(features) for layer, features in zip(self.landmark_heads, shared)],
            )
        return outputs


class YuNet(nn.Module):
    """Randomly initialized YuNet-N; box-only by default."""

    strides = (8, 16, 32)

    def __init__(self, *, landmarks: bool = False):
        super().__init__()
        self.backbone = _YuNetNBackbone()
        self.neck = _TFPN()
        self.landmarks = landmarks
        self.head = _BoxOnlyHead(landmarks=landmarks)

    def forward(self, inputs: torch.Tensor) -> YuNetOutputs | YuNetLandmarkOutputs:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError(
                f"Expected RGB input shaped [N, 3, H, W], got {tuple(inputs.shape)}"
            )
        if inputs.shape[2] % 32 or inputs.shape[3] % 32:
            raise ValueError("Input height and width must both be divisible by 32")
        return self.head(self.neck(self.backbone(inputs)))


def build_yunet_n(*, landmarks: bool = False) -> YuNet:
    """Build YuNet-N with an optional five-point training head."""

    return YuNet(landmarks=landmarks)
