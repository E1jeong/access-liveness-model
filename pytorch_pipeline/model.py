import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torchvision.models as models
from classes import CLASS_NAMES

def replace_hardswish_with_relu(model):
    """NPU 호환성을 위해 Hardswish를 ReLU6로 치환합니다."""
    for name, child in model.named_children():
        if isinstance(child, nn.Hardswish):
            setattr(model, name, nn.ReLU6())
        else:
            replace_hardswish_with_relu(child)

def disable_se_blocks(model):
    """NPU 연산자 제약 및 가속 병목 방지를 위해 SE(Squeeze-and-Excitation) 블록을 Identity로 비활성화합니다."""
    from torchvision.ops.misc import SqueezeExcitation
    for name, child in model.named_children():
        if isinstance(child, SqueezeExcitation):
            setattr(model, name, nn.Identity())
        else:
            disable_se_blocks(child)

def _build_npu_classifier_head(in_channels, num_classes, hidden_units=1024, dropout=0.2):
    """NPU 친화적 1x1 Conv2D 기반 분류기 헤드 구성 (Flatten/Linear 미사용)"""
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_units, kernel_size=1),
        nn.ReLU6(),
        nn.Dropout(p=dropout),
        nn.Conv2d(hidden_units, num_classes, kernel_size=1)
    )

class SingleInputMobileNetV3(nn.Module):
    """1채널 IR 또는 3채널 RGB 단일 입력을 위한 NPU 친화적 MobileNetV3-Small 모델"""
    def __init__(self, in_channels=1, num_classes=len(CLASS_NAMES), conv1_reduction="sum", pretrained=True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_model = models.mobilenet_v3_small(weights=weights)
        replace_hardswish_with_relu(base_model)
        disable_se_blocks(base_model)

        if in_channels == 1:
            first_conv = base_model.features[0][0]
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )
            if pretrained and first_conv.weight is not None:
                with torch.no_grad():
                    if conv1_reduction == "sum":
                        new_conv.weight.data = first_conv.weight.data.sum(dim=1, keepdim=True)
                    else:
                        new_conv.weight.data = first_conv.weight.data.mean(dim=1, keepdim=True)
            base_model.features[0][0] = new_conv

        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = _build_npu_classifier_head(576, num_classes)

    def forward(self, x):
        # x: [B, C, 224, 224]
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x.flatten(1)

class DualInputMobileNetV3(nn.Module):
    """RGB + IR 듀얼 입력을 위한 NPU 친화적 MobileNetV3-Small 모델"""
    def __init__(self, num_classes=len(CLASS_NAMES), conv1_reduction="sum", pretrained=True):
        super().__init__()
        # RGB Backbone (MobileNetV3-Small)
        rgb_weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        rgb_model = models.mobilenet_v3_small(weights=rgb_weights)
        replace_hardswish_with_relu(rgb_model)
        disable_se_blocks(rgb_model)
        self.rgb_features = rgb_model.features
        self.rgb_pool = nn.AdaptiveAvgPool2d((1, 1))

        # IR Backbone (MobileNetV3-Small)
        ir_weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        ir_model = models.mobilenet_v3_small(weights=ir_weights)
        replace_hardswish_with_relu(ir_model)
        disable_se_blocks(ir_model)
        first_conv = ir_model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        if pretrained and first_conv.weight is not None:
            with torch.no_grad():
                if conv1_reduction == "sum":
                    new_conv.weight.data = first_conv.weight.data.sum(dim=1, keepdim=True)
                else:
                    new_conv.weight.data = first_conv.weight.data.mean(dim=1, keepdim=True)
        ir_model.features[0][0] = new_conv
        self.ir_features = ir_model.features
        self.ir_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final 1x1 Conv classifier layer: 576 (RGB) + 576 (IR) = 1152 features
        self.classifier = _build_npu_classifier_head(1152, num_classes)

    def forward(self, rgb, ir):
        # rgb: [B, 3, 224, 224], ir: [B, 1, 224, 224] (in PyTorch NCHW layout)
        f_rgb = self.rgb_pool(self.rgb_features(rgb))
        f_ir = self.ir_pool(self.ir_features(ir))
        f_fused = torch.cat((f_rgb, f_ir), dim=1)
        out = self.classifier(f_fused)
        return out.flatten(1)

def get_anti_spoof_model(model_type="crop_ir", num_classes=len(CLASS_NAMES), conv1_reduction="sum", pretrained=True):
    """
    안드로이드 기기/NPU에 적합한 NPU 친화적 MobileNetV3-Small 모델을 빌드합니다.
    - model_type: 'crop_ir', 'crop_rgb', 'dual'
    """
    if model_type in ("crop_ir", "single_ir"):
        model = SingleInputMobileNetV3(in_channels=1, num_classes=num_classes, conv1_reduction=conv1_reduction, pretrained=pretrained)
    elif model_type in ("crop_rgb", "single_rgb"):
        model = SingleInputMobileNetV3(in_channels=3, num_classes=num_classes, conv1_reduction=conv1_reduction, pretrained=pretrained)
    elif model_type in ("dual", "dual_input"):
        model = DualInputMobileNetV3(num_classes=num_classes, conv1_reduction=conv1_reduction, pretrained=pretrained)
    else:
        raise ValueError(f"지원하지 않는 model_type: {model_type}")

    print(f"[모델 생성 완료]")
    print(f" - 모델 타입: {model_type}")
    print(f" - 분류 클래스 수: {num_classes} ({', '.join(f'{idx}={name}' for idx, name in enumerate(CLASS_NAMES[:num_classes]))})")
    print(f" - Conv1 축소 방식: {conv1_reduction}")
    return model

if __name__ == "__main__":
    model = get_anti_spoof_model(model_type="crop_ir", num_classes=len(CLASS_NAMES))
    dummy_ir = torch.randn(1, 1, 224, 224)
    output = model(dummy_ir)
    print(f"Single IR 모델 출력 크기: {output.shape}")

    dual_model = get_anti_spoof_model(model_type="dual", num_classes=len(CLASS_NAMES))
    dummy_rgb = torch.randn(1, 3, 224, 224)
    output_dual = dual_model(dummy_rgb, dummy_ir)
    print(f"Dual 모델 출력 크기: {output_dual.shape}")
