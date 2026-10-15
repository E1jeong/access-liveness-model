import pytest
import torch

from face_detection_pipeline.evaluation.detection import (
    Detection,
    calculate_detection_metrics,
    decode_detections,
)
from face_detection_pipeline.models.yunet import YuNetOutputs


def test_decode_detections_restores_box_and_score():
    outputs = YuNetOutputs(
        cls_scores=[torch.tensor([[[[10.0]]]])],
        objectnesses=[torch.tensor([[[[10.0]]]])],
        bbox_regressions=[torch.tensor([[[[0.5]], [[0.5]], [[0.0]], [[0.0]]]])],
    )

    detections = decode_detections(outputs, strides=(8,))

    assert len(detections) == 1
    assert torch.allclose(detections[0].boxes, torch.tensor([[0.0, 0.0, 8.0, 8.0]]))
    assert detections[0].scores.item() == pytest.approx(0.999909, rel=1e-5)


def test_decode_detections_suppresses_overlapping_lower_score_box():
    outputs = YuNetOutputs(
        cls_scores=[torch.tensor([[[[10.0, 9.0]]]])],
        objectnesses=[torch.tensor([[[[10.0, 10.0]]]])],
        bbox_regressions=[
            torch.tensor(
                [[[[0.5, -0.5]], [[0.5, 0.5]], [[0.0, 0.0]], [[0.0, 0.0]]]]
            )
        ],
    )

    detections = decode_detections(outputs, strides=(8,))

    assert detections[0].boxes.shape == (1, 4)


def test_detection_metrics_count_matches_and_geometry_errors():
    detections = [
        Detection(
            boxes=torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
            scores=torch.tensor([0.9, 0.8]),
        ),
        Detection(boxes=torch.empty((0, 4)), scores=torch.empty((0,))),
    ]
    ground_truth = [
        torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        torch.tensor([[5.0, 5.0, 15.0, 15.0]]),
    ]

    metrics = calculate_detection_metrics(detections, ground_truth)

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["false_positives_per_image"] == pytest.approx(0.5)
    assert metrics["mean_iou"] == pytest.approx(1.0)
    assert metrics["mean_center_error"] == pytest.approx(0.0)
    assert metrics["mean_scale_error"] == pytest.approx(0.0)


def test_detection_metrics_reject_mismatched_batches():
    with pytest.raises(ValueError):
        calculate_detection_metrics([], [torch.empty((0, 4))])
