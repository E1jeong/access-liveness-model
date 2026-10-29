import pytest
import torch
import torch.nn as nn
from torchvision.ops.misc import SqueezeExcitation

from common.classes import CLASS_NAMES
from antispoofing_pipeline.pytorch.model import (
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

@pytest.mark.parametrize("model_type", ["crop_ir", "dual"])
def test_pytorch_conv1_reduction(model_type):
    from torchvision import models

    source = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model_sum = get_anti_spoof_model(model_type=model_type, pretrained=True)
    features = model_sum.ir_features if model_type == "dual" else model_sum.features
    first_conv = features[0][0]
    assert first_conv.in_channels == 1
    assert first_conv.weight.shape == (16, 1, 3, 3)
    torch.testing.assert_close(first_conv.weight, source.features[0][0].weight.sum(dim=1, keepdim=True))


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_pytorch_train_rejects_removed_conv1_option(monkeypatch, reduction):
    from antispoofing_pipeline.pytorch.train import parse_args

    monkeypatch.setattr("sys.argv", ["train.py", "--conv1-reduction", reduction])
    with pytest.raises(SystemExit):
        parse_args()

    monkeypatch.setattr("sys.argv", ["train.py"])
    assert not hasattr(parse_args(), "conv1_reduction")
