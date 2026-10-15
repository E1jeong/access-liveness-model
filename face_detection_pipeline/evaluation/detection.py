"""Decode YuNet outputs and calculate box-level detection metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from face_detection_pipeline.models.yunet import YuNetOutputs


@dataclass(frozen=True)
class Detection:
    boxes: torch.Tensor
    scores: torch.Tensor


def decode_detections(
    outputs: YuNetOutputs,
    *,
    strides: tuple[int, ...] = (8, 16, 32),
    score_threshold: float = 0.5,
    nms_threshold: float = 0.45,
    max_detections: int = 100,
) -> list[Detection]:
    """Decode raw outputs into letterboxed-image coordinates."""

    if not 0 <= score_threshold <= 1 or not 0 <= nms_threshold <= 1:
        raise ValueError("score and NMS thresholds must be in [0, 1]")
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")
    if len(outputs.cls_scores) != len(strides):
        raise ValueError("Expected one stride for each prediction level")

    batch_size = outputs.cls_scores[0].shape[0]
    feature_sizes = [tuple(scores.shape[-2:]) for scores in outputs.cls_scores]
    priors = _grid_priors(
        feature_sizes,
        strides,
        dtype=outputs.cls_scores[0].dtype,
        device=outputs.cls_scores[0].device,
    )
    cls_scores = _flatten(outputs.cls_scores, batch_size, 1).sigmoid()
    objectnesses = _flatten(outputs.objectnesses, batch_size, 1).sigmoid()
    regressions = _flatten(outputs.bbox_regressions, batch_size, 4)
    decoded = _decode_boxes(priors.unsqueeze(0), regressions)
    scores = (cls_scores * objectnesses).squeeze(-1)
    image_height = max(
        height * stride for (height, _), stride in zip(feature_sizes, strides)
    )
    image_width = max(
        width * stride for (_, width), stride in zip(feature_sizes, strides)
    )

    detections = []
    for image_index in range(batch_size):
        selected = scores[image_index] >= score_threshold
        boxes = decoded[image_index, selected].clone()
        selected_scores = scores[image_index, selected]
        if boxes.numel():
            boxes[:, 0::2].clamp_(0, image_width)
            boxes[:, 1::2].clamp_(0, image_height)
            keep = _nms(boxes, selected_scores, nms_threshold)[:max_detections]
            boxes = boxes[keep]
            selected_scores = selected_scores[keep]
        detections.append(Detection(boxes=boxes.reshape(-1, 4), scores=selected_scores))
    return detections


def calculate_detection_metrics(
    detections: list[Detection],
    ground_truth_boxes: list[torch.Tensor],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    if len(detections) != len(ground_truth_boxes):
        raise ValueError("Expected one detection and ground-truth entry per image")
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    matched_ious = []
    center_errors = []
    scale_errors = []
    for detection, ground_truth in zip(detections, ground_truth_boxes):
        predicted = detection.boxes.detach().cpu().float().reshape(-1, 4)
        scores = detection.scores.detach().cpu().float().reshape(-1)
        target = ground_truth.detach().cpu().float().reshape(-1, 4)
        if predicted.numel() == 0:
            false_negatives += target.shape[0]
            continue
        if target.numel() == 0:
            false_positives += predicted.shape[0]
            continue

        overlaps = _box_iou(predicted, target)
        used_targets: set[int] = set()
        for prediction_index in scores.argsort(descending=True).tolist():
            available_overlaps = overlaps[prediction_index].clone()
            if used_targets:
                available_overlaps[list(used_targets)] = -1
            best_iou, target_index = available_overlaps.max(dim=0)
            target_index_value = int(target_index)
            if best_iou >= iou_threshold:
                used_targets.add(target_index_value)
                true_positives += 1
                matched_ious.append(float(best_iou))
                center_error, scale_error = _geometry_errors(
                    predicted[prediction_index],
                    target[target_index_value],
                )
                center_errors.append(center_error)
                scale_errors.append(scale_error)
            else:
                false_positives += 1
        false_negatives += target.shape[0] - len(used_targets)

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    image_count = max(len(detections), 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives_per_image": false_positives / image_count,
        "mean_iou": sum(matched_ious) / max(len(matched_ious), 1),
        "mean_center_error": sum(center_errors) / max(len(center_errors), 1),
        "mean_scale_error": sum(scale_errors) / max(len(scale_errors), 1),
    }


def _flatten(
    tensors: list[torch.Tensor],
    batch_size: int,
    channels: int,
) -> torch.Tensor:
    return torch.cat(
        [
            tensor.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
            for tensor in tensors
        ],
        dim=1,
    )


def _grid_priors(
    feature_sizes: list[tuple[int, int]],
    strides: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    levels = []
    for (height, width), stride in zip(feature_sizes, strides):
        shifts_x = torch.arange(width, dtype=dtype, device=device) * stride
        shifts_y = torch.arange(height, dtype=dtype, device=device) * stride
        grid_y, grid_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        stride_values = grid_x.new_full((height * width,), stride)
        levels.append(
            torch.stack(
                [
                    grid_x.reshape(-1),
                    grid_y.reshape(-1),
                    stride_values,
                    stride_values,
                ],
                dim=-1,
            )
        )
    return torch.cat(levels)


def _decode_boxes(priors: torch.Tensor, regressions: torch.Tensor) -> torch.Tensor:
    centers = regressions[..., :2] * priors[..., 2:] + priors[..., :2]
    sizes = regressions[..., 2:].clamp(-16, 16).exp() * priors[..., 2:]
    return torch.cat([centers - sizes / 2, centers + sizes / 2], dim=-1)


def _nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        overlaps = _box_iou(boxes[current].reshape(1, 4), boxes[remaining]).squeeze(0)
        order = remaining[overlaps <= threshold]
    if not keep:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    return torch.stack(keep)


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    intersection_size = (bottom_right - top_left).clamp(min=0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    return intersection / (area1[:, None] + area2 - intersection + 1e-7)


def _geometry_errors(predicted: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    predicted_size = predicted[2:] - predicted[:2]
    target_size = target[2:] - target[:2]
    predicted_center = (predicted[:2] + predicted[2:]) / 2
    target_center = (target[:2] + target[2:]) / 2
    target_diagonal = torch.linalg.vector_norm(target_size).clamp_min(1e-7)
    center_error = torch.linalg.vector_norm(predicted_center - target_center) / target_diagonal
    scale_error = torch.log((predicted_size / target_size).clamp_min(1e-7)).abs().mean()
    return float(center_error), float(scale_error)
