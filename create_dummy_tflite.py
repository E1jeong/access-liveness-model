import torch
import torch.nn as nn
import torchvision.models as models
import litert_torch
import os

class DualInputMobileNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        # RGB Backbone (MobileNetV3-Small)
        rgb_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.rgb_features = rgb_model.features
        self.rgb_pool = rgb_model.avgpool
        
        # IR Backbone (MobileNetV3-Small)
        ir_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Modify first conv layer of IR backbone to accept 1 channel instead of 3
        first_conv = ir_model.features[0][0]
        ir_model.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
        self.ir_features = ir_model.features
        self.ir_pool = ir_model.avgpool
        
        # Final linear classifier layer
        # Output features: 576 (RGB) + 576 (IR) = 1152 features
        self.classifier = nn.Sequential(
            nn.Linear(1152, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, 4)
        )

    def forward(self, rgb, ir):
        # rgb: [1, 3, 224, 224], ir: [1, 1, 224, 224] (in PyTorch NCHW layout)
        f_rgb = self.rgb_features(rgb) # [1, 576, 7, 7]
        f_rgb = self.rgb_pool(f_rgb) # [1, 576, 1, 1]
        f_rgb = torch.flatten(f_rgb, 1) # [1, 576]
        
        f_ir = self.ir_features(ir) # [1, 576, 7, 7]
        f_ir = self.ir_pool(f_ir) # [1, 576, 1, 1]
        f_ir = torch.flatten(f_ir, 1) # [1, 576]
        
        # Concatenate features
        f_fused = torch.cat((f_rgb, f_ir), dim=1) # [1, 1152]
        
        # Classify
        out = self.classifier(f_fused) # [1, 4]
        return out

def main():
    model = DualInputMobileNetV3()
    model.eval()

    # Define dummy NHWC sample inputs (since we wrap the model with to_channel_last_io)
    sample_rgb = torch.randn(1, 224, 224, 3)
    sample_ir = torch.randn(1, 224, 224, 1)
    
    # Wrap model to accept and produce NHWC (channels last) format
    # Since we have two inputs at index 0 and 1
    nhwc_model = litert_torch.to_channel_last_io(model, args=[0, 1])
    nhwc_model.eval()

    # Convert to TFLite using litert_torch
    print("Converting dual-input MobileNetV3-Small model to TFLite (NHWC format)...")
    edge_model = litert_torch.convert(nhwc_model, (sample_rgb, sample_ir))
    
    tflite_path = "anti_spoofing.tflite"
    edge_model.export(tflite_path)
    print(f"[Success] Exported realistic model to {tflite_path}")

if __name__ == "__main__":
    main()
