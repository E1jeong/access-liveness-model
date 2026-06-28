import torch
import torch.nn as nn
import torchvision.models as models
from classes import CLASS_NAMES

class DualInputMobileNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        # RGB Backbone (MobileNetV3-Small)
        rgb_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.rgb_features = rgb_model.features
        self.rgb_pool = rgb_model.avgpool

        # IR Backbone (MobileNetV3-Small)
        ir_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        first_conv = ir_model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        # 3채널 pretrained 가중치를 채널 방향 평균으로 1채널에 이전
        with torch.no_grad():
            new_conv.weight.data = first_conv.weight.data.mean(dim=1, keepdim=True)
        ir_model.features[0][0] = new_conv
        self.ir_features = ir_model.features
        self.ir_pool = ir_model.avgpool

        # Final linear classifier layer
        # Output features: 576 (RGB) + 576 (IR) = 1152 features
        self.classifier = nn.Sequential(
            nn.Linear(1152, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, len(CLASS_NAMES))
        )

    def forward(self, rgb, ir):
        # rgb: [B, 3, 224, 224], ir: [B, 1, 224, 224] (in PyTorch NCHW layout)
        f_rgb = self.rgb_features(rgb)   # [B, 576, 7, 7]
        f_rgb = self.rgb_pool(f_rgb)     # [B, 576, 1, 1]
        f_rgb = torch.flatten(f_rgb, 1)  # [B, 576]

        f_ir = self.ir_features(ir)      # [B, 576, 7, 7]
        f_ir = self.ir_pool(f_ir)        # [B, 576, 1, 1]
        f_ir = torch.flatten(f_ir, 1)    # [B, 576]

        f_fused = torch.cat((f_rgb, f_ir), dim=1)  # [B, 1152]
        return self.classifier(f_fused)             # [B, 5]

def get_anti_spoof_model():
    """
    안드로이드 기기/NPU에 적합한 듀얼 인풋(RGB + IR) MobileNetV3-Small 모델을 빌드합니다.
    """
    model = DualInputMobileNetV3()
    print("[모델 생성 완료]")
    print(f" - 베이스 모델: Dual-Input MobileNetV3-Small")
    print(f" - 분류 클래스 수: {len(CLASS_NAMES)} (0=live, 1=print, 2=picture, 3=mask, 4=display)")
    return model

if __name__ == "__main__":
    model = get_anti_spoof_model()
    dummy_rgb = torch.randn(1, 3, 224, 224)
    dummy_ir = torch.randn(1, 1, 224, 224)
    output = model(dummy_rgb, dummy_ir)
    print(f"모델 출력 텐서 크기: {output.shape}")  # [1, 5]
    print(f"예측 출력값: {output}")
