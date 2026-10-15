import pytest
import torch

from face_detection_pipeline.models.yunet import YuNetOutputs, build_yunet_n


@pytest.mark.parametrize(
    ("height", "width", "feature_shapes"),
    [
        (320, 320, [(40, 40), (20, 20), (10, 10)]),
        (448, 256, [(56, 32), (28, 16), (14, 8)]),
    ],
)
def test_yunet_n_box_only_output_shapes(height, width, feature_shapes):
    model = build_yunet_n().eval()

    with torch.inference_mode():
        outputs = model(torch.randn(1, 3, height, width))

    assert isinstance(outputs, YuNetOutputs)
    assert len(outputs.cls_scores) == 3
    assert len(outputs.objectnesses) == 3
    assert len(outputs.bbox_regressions) == 3
    for level, (feature_height, feature_width) in enumerate(feature_shapes):
        assert outputs.cls_scores[level].shape == (1, 1, feature_height, feature_width)
        assert outputs.objectnesses[level].shape == (
            1,
            1,
            feature_height,
            feature_width,
        )
        assert outputs.bbox_regressions[level].shape == (
            1,
            4,
            feature_height,
            feature_width,
        )


def test_yunet_n_has_no_landmark_head():
    model = build_yunet_n()

    assert sum(parameter.numel() for parameter in model.parameters()) == 73_606
    assert all(
        "kps" not in name and "landmark" not in name for name, _ in model.named_modules()
    )


@pytest.mark.parametrize("shape", [(1, 1, 320, 320), (1, 3, 321, 320)])
def test_yunet_n_rejects_unsupported_input_shape(shape):
    model = build_yunet_n()

    with pytest.raises(ValueError):
        model(torch.randn(shape))
