import pytest
import torch
import torch.nn as nn
from torchvision.ops.misc import SqueezeExcitation

from classes import CLASS_NAMES
from pytorch_pipeline.model import (
    get_anti_spoof_model,
    SingleInputMobileNetV3,
    DualInputMobileNetV3,
)

def test_pytorch_single_ir_model_output_shape():
    model = get_anti_spoof_model(model_type="crop_ir", num_classes=len(CLASS_NAMES), pretrained=False)
    dummy_ir = torch.randn(2, 1, 224, 224)
    out = model(dummy_ir)
    assert out.shape == (2, len(CLASS_NAMES))
    assert out.dtype == torch.float32

def test_pytorch_dual_model_output_shape():
    model = get_anti_spoof_model(model_type="dual", num_classes=len(CLASS_NAMES), pretrained=False)
    dummy_rgb = torch.randn(2, 3, 224, 224)
    dummy_ir = torch.randn(2, 1, 224, 224)
    out = model(dummy_rgb, dummy_ir)
    assert out.shape == (2, len(CLASS_NAMES))
    assert out.dtype == torch.float32

def test_pytorch_model_npu_friendly_ops():
    model = get_anti_spoof_model(model_type="crop_ir", pretrained=False)
    # 1. No Hardswish
    for name, mod in model.named_modules():
        assert not isinstance(mod, nn.Hardswish), f"Found Hardswish at {name}"
        assert not isinstance(mod, SqueezeExcitation), f"Found SqueezeExcitation at {name}"
        # Linear layer should not be in classifier head
        if "classifier" in name:
            assert not isinstance(mod, nn.Linear), f"Found Linear layer in classifier at {name}"

def test_pytorch_conv1_reduction():
    model_sum = get_anti_spoof_model(model_type="crop_ir", conv1_reduction="sum", pretrained=True)
    first_conv = model_sum.features[0][0]
    assert first_conv.in_channels == 1
    assert first_conv.weight.shape == (16, 1, 3, 3)
