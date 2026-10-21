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


def test_decode_landmarks_uses_grid_stride_and_same_nms_indices():
    from face_detection_pipeline.models.yunet import YuNetLandmarkOutputs

    outputs = YuNetLandmarkOutputs(
        [torch.tensor([[[[9., 10.]]]])], [torch.full((1, 1, 1, 2), 10.)],
        [torch.tensor([[[[0.5, -0.5]], [[0.5, 0.5]], [[0., 0.]], [[0., 0.]]]])],
        [torch.tensor([[[[i, i + 1.]] for i in range(10)]])],
    )
    detection = decode_detections(outputs, strides=(8,))[0]
    assert detection.landmarks.shape == (1, 5, 2)
    expected = torch.arange(1., 11.).reshape(1, 5, 2) * 8
    expected[:, :, 0] += 8
    torch.testing.assert_close(detection.landmarks, expected)
    assert decode_detections(outputs, strides=(8,), score_threshold=1)[0].landmarks.shape == (0, 5, 2)


def test_landmark_restore_letterbox_and_masked_metrics():
    from PIL import Image
    from face_detection_pipeline.data.dataset import letterbox_image_and_boxes, restore_landmarks_from_letterbox
    from face_detection_pipeline.evaluation.detection import calculate_landmark_metrics

    _, boxes, geometry = letterbox_image_and_boxes(Image.new('RGB', (96, 128)),
                                                   torch.tensor([[20., 20., 75., 110.]]), (64, 64))
    points = torch.tensor([[[23., 20.]] * 5])
    restored = restore_landmarks_from_letterbox(points, geometry)
    torch.testing.assert_close(restored[0, 0], torch.tensor([30., 40.]))
    detection = Detection(boxes, torch.tensor([0.9]), points)
    targets = points.clone()
    targets[:, 0] += torch.tensor([3., 4.])
    masks = [torch.tensor([[True, False, False, False, False]])]
    metrics = calculate_landmark_metrics([detection], [boxes], [targets], masks)
    assert metrics['mean_error_pixels'] == pytest.approx(5.)
    assert metrics['matched_points'] == 1
    assert metrics['target_points'] == 1
