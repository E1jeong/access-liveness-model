import pytest
import torch

from face_detection_pipeline.models.yunet import YuNetOutputs, build_yunet_n
from face_detection_pipeline.training.loss import YuNetCriterion


@pytest.mark.parametrize(
    "boxes",
    [
        [torch.tensor([[12.0, 10.0, 52.0, 56.0]])],
        [torch.empty((0, 4))],
    ],
)
def test_box_only_loss_is_finite_and_backpropagates(boxes):
    model = build_yunet_n()
    outputs = model(torch.randn(1, 3, 64, 64))

    losses = YuNetCriterion()(outputs, boxes)

    assert set(losses) == {"total", "classification", "objectness", "bbox"}
    assert all(torch.isfinite(loss) for loss in losses.values())
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_box_only_loss_rejects_batch_target_count_mismatch():
    outputs = build_yunet_n()(torch.randn(2, 3, 64, 64))

    with pytest.raises(ValueError):
        YuNetCriterion()(outputs, [torch.empty((0, 4))])


def test_box_only_loss_matches_pinned_upstream_reference_values():
    torch.manual_seed(7)
    feature_sizes = [(4, 4), (2, 2), (1, 1)]
    cls_scores = [torch.randn(1, 1, h, w) for h, w in feature_sizes]
    objectnesses = [torch.randn(1, 1, h, w) for h, w in feature_sizes]
    bbox_regressions = [
        torch.randn(1, 4, h, w) * 0.1 for h, w in feature_sizes
    ]
    outputs = YuNetOutputs(cls_scores, objectnesses, bbox_regressions)

    losses = YuNetCriterion()(
        outputs,
        [torch.tensor([[4.0, 4.0, 24.0, 24.0]])],
    )

    assert losses["classification"].item() == pytest.approx(0.4770355821)
    assert losses["objectness"].item() == pytest.approx(18.3042316437)
    assert losses["bbox"].item() == pytest.approx(3.8692531586)
