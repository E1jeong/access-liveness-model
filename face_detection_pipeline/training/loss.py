"""YuNet assignment and box/optional masked landmark loss.

Derived from Shiqi Yu's ``libfacedetection.train`` at commit
``dca340aa082c71081a68d17db8e58b33a58a914b``. The upstream project is
licensed under BSD-3-Clause; see ``../third_party/yunet/LICENSE``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from face_detection_pipeline.models.yunet import YuNetLandmarkOutputs, YuNetOutputs


@dataclass(frozen=True)
class YuNetLossWeights:
    classification: float = 1.0
    objectness: float = 1.0
    bbox: float = 5.0
    landmark: float = 1.0


@dataclass(frozen=True)
class _AssignResult:
    gt_indices: torch.Tensor
    overlaps: torch.Tensor


class _PointGenerator:
    def __init__(self, strides: tuple[int, ...]):
        self.strides = strides

    def grid_priors(
        self,
        feature_sizes: list[tuple[int, int]],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if len(feature_sizes) != len(self.strides):
            raise ValueError(
                f"Expected {len(self.strides)} feature maps, got {len(feature_sizes)}"
            )
        levels = []
        for (height, width), stride in zip(feature_sizes, self.strides):
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


class _SimOTAAssigner:
    def __init__(self, center_radius: float = 2.5, candidate_topk: int = 10):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk

    @torch.no_grad()
    def assign(
        self,
        scores: torch.Tensor,
        priors: torch.Tensor,
        decoded_boxes: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> _AssignResult:
        assigned = torch.zeros(
            decoded_boxes.shape[0],
            dtype=torch.long,
            device=decoded_boxes.device,
        )
        overlaps = decoded_boxes.new_zeros((decoded_boxes.shape[0],))
        if ground_truth.numel() == 0 or decoded_boxes.numel() == 0:
            return _AssignResult(assigned, overlaps)

        valid_mask, in_box_and_center = self._candidate_mask(priors, ground_truth)
        if not valid_mask.any():
            return _AssignResult(assigned, overlaps)

        valid_boxes = decoded_boxes[valid_mask]
        valid_scores = scores[valid_mask]
        pairwise_ious = _bbox_overlaps(valid_boxes, ground_truth)
        iou_cost = -torch.log(pairwise_ious + 1e-7)
        classification_cost = F.binary_cross_entropy(
            valid_scores[:, None, :].float().sqrt().expand(-1, ground_truth.shape[0], -1),
            torch.ones(
                (valid_scores.shape[0], ground_truth.shape[0], 1),
                dtype=torch.float32,
                device=valid_scores.device,
            ),
            reduction="none",
        ).sum(-1)
        cost = classification_cost + 3.0 * iou_cost + (~in_box_and_center) * 100_000.0
        matched_ious, matched_gt, matched_valid = self._dynamic_match(
            cost,
            pairwise_ious,
        )

        foreground = valid_mask.clone()
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
        foreground[valid_indices] = False
        foreground[valid_indices[matched_valid]] = True
        assigned[foreground] = matched_gt + 1
        overlaps[foreground] = matched_ious
        return _AssignResult(assigned, overlaps)

    def _candidate_mask(
        self,
        priors: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centers_x = priors[:, 0, None]
        centers_y = priors[:, 1, None]
        stride_x = priors[:, 2, None]
        stride_y = priors[:, 3, None]

        in_boxes = torch.stack(
            [
                centers_x - ground_truth[:, 0],
                centers_y - ground_truth[:, 1],
                ground_truth[:, 2] - centers_x,
                ground_truth[:, 3] - centers_y,
            ],
            dim=1,
        ).amin(dim=1) > 0

        gt_centers_x = (ground_truth[:, 0] + ground_truth[:, 2]) / 2
        gt_centers_y = (ground_truth[:, 1] + ground_truth[:, 3]) / 2
        in_centers = torch.stack(
            [
                centers_x - (gt_centers_x - self.center_radius * stride_x),
                centers_y - (gt_centers_y - self.center_radius * stride_y),
                (gt_centers_x + self.center_radius * stride_x) - centers_x,
                (gt_centers_y + self.center_radius * stride_y) - centers_y,
            ],
            dim=1,
        ).amin(dim=1) > 0

        valid = in_boxes.any(dim=1) | in_centers.any(dim=1)
        return valid, in_boxes[valid] & in_centers[valid]

    def _dynamic_match(
        self,
        cost: torch.Tensor,
        pairwise_ious: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        matching = torch.zeros_like(cost, dtype=torch.bool)
        topk = min(self.candidate_topk, pairwise_ious.shape[0])
        dynamic_counts = torch.clamp(
            torch.topk(pairwise_ious, topk, dim=0).values.sum(0).int(),
            min=1,
        )
        for gt_index in range(pairwise_ious.shape[1]):
            count = int(dynamic_counts[gt_index].item())
            selected = torch.topk(cost[:, gt_index], count, largest=False).indices
            matching[selected, gt_index] = True

        multiply_matched = matching.sum(dim=1) > 1
        if multiply_matched.any():
            best_gt = cost[multiply_matched].argmin(dim=1)
            matching[multiply_matched] = False
            matching[multiply_matched, best_gt] = True

        foreground = matching.any(dim=1)
        matched_gt = matching[foreground].float().argmax(dim=1)
        matched_ious = (
            matching[foreground] * pairwise_ious[foreground]
        ).sum(dim=1)
        return matched_ious, matched_gt, foreground


class YuNetCriterion:
    """BCE/EIoU detection loss and optional point-masked Smooth L1 in stride units."""

    def __init__(
        self,
        *,
        strides: tuple[int, ...] = (8, 16, 32),
        weights: YuNetLossWeights | None = None,
    ):
        self.prior_generator = _PointGenerator(strides)
        self.assigner = _SimOTAAssigner()
        self.weights = weights or YuNetLossWeights()

    def __call__(
        self,
        outputs: YuNetOutputs | YuNetLandmarkOutputs,
        ground_truth_boxes: list[torch.Tensor],
        ground_truth_landmarks: list[torch.Tensor] | None = None,
        landmark_masks: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = outputs.cls_scores[0].shape[0]
        if len(ground_truth_boxes) != batch_size:
            raise ValueError("Expected one ground-truth box tensor per image")
        landmark_outputs = getattr(outputs, "landmark_regressions", None)
        if landmark_outputs is not None:
            if (ground_truth_landmarks is None or landmark_masks is None
                    or len(ground_truth_landmarks) != batch_size
                    or len(landmark_masks) != batch_size):
                raise ValueError("Expected landmark targets and masks per image")
            for boxes, points, mask in zip(
                ground_truth_boxes, ground_truth_landmarks, landmark_masks,
            ):
                if (points.shape != (boxes.shape[0], 5, 2)
                        or mask.shape != (boxes.shape[0], 5) or mask.dtype != torch.bool):
                    raise ValueError("Invalid landmark target shape or mask dtype")
                if not torch.isfinite(points[mask]).all():
                    raise ValueError("Selected landmark targets must be finite")
        elif ground_truth_landmarks is not None or landmark_masks is not None:
            raise ValueError("Landmark targets require a landmark model")

        feature_sizes = [tuple(scores.shape[-2:]) for scores in outputs.cls_scores]
        priors = self.prior_generator.grid_priors(
            feature_sizes,
            dtype=outputs.cls_scores[0].dtype,
            device=outputs.cls_scores[0].device,
        )
        cls_logits = _flatten(outputs.cls_scores, batch_size, 1)
        objectness_logits = _flatten(outputs.objectnesses, batch_size, 1)
        bbox_regressions = _flatten(outputs.bbox_regressions, batch_size, 4)
        expanded_priors = priors.unsqueeze(0).expand(batch_size, -1, -1)
        decoded_boxes = _bbox_decode(expanded_priors, bbox_regressions)
        assignment_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]],
            dim=-1,
        )

        cls_targets = []
        objectness_targets = []
        bbox_targets = []
        positive_masks = []
        positive_count = 0
        landmark_targets = []
        selected_landmark_masks = []
        for image_index, boxes in enumerate(ground_truth_boxes):
            boxes = boxes.to(device=cls_logits.device, dtype=cls_logits.dtype).reshape(-1, 4)
            assignment = self.assigner.assign(
                cls_logits[image_index].detach().sigmoid()
                * objectness_logits[image_index].detach().sigmoid(),
                assignment_priors,
                decoded_boxes[image_index].detach(),
                boxes,
            )
            positive = assignment.gt_indices > 0
            assigned_indices = assignment.gt_indices[positive] - 1
            positive_masks.append(positive)
            positive_count += int(positive.sum())
            cls_targets.append(assignment.overlaps[positive, None])
            objectness_targets.append(positive.to(cls_logits.dtype)[:, None])
            bbox_targets.append(boxes[assigned_indices])
            if landmark_outputs is not None:
                points = ground_truth_landmarks[image_index].to(
                    device=cls_logits.device, dtype=cls_logits.dtype,
                )[assigned_indices]
                mask = landmark_masks[image_index].to(device=cls_logits.device)[assigned_indices]
                # Encode in stride units using the same grid origin as box decoding.
                target = (points - priors[positive, None, :2]) / priors[positive, None, 2:]
                landmark_targets.append(target)
                selected_landmark_masks.append(mask)

        normalizer = max(float(positive_count), 1.0)
        positive_mask = torch.cat(positive_masks)
        flat_cls = cls_logits.reshape(-1, 1)
        flat_objectness = objectness_logits.reshape(-1, 1)
        flat_boxes = decoded_boxes.reshape(-1, 4)
        objectness_target = torch.cat(objectness_targets)

        loss_objectness = F.binary_cross_entropy_with_logits(
            flat_objectness,
            objectness_target,
            reduction="sum",
        ) / normalizer
        if positive_count:
            loss_classification = F.binary_cross_entropy_with_logits(
                flat_cls[positive_mask],
                torch.cat(cls_targets),
                reduction="sum",
            ) / normalizer
            loss_bbox = _eiou_loss(
                flat_boxes[positive_mask],
                torch.cat(bbox_targets),
            ).sum() / normalizer
        else:
            zero = flat_cls.sum() * 0
            loss_classification = zero
            loss_bbox = zero

        weighted_classification = loss_classification * self.weights.classification
        weighted_objectness = loss_objectness * self.weights.objectness
        weighted_bbox = loss_bbox * self.weights.bbox
        losses = {
            "total": weighted_classification + weighted_objectness + weighted_bbox,
            "classification": weighted_classification,
            "objectness": weighted_objectness,
            "bbox": weighted_bbox,
        }
        if landmark_outputs is not None:
            predictions = _flatten(landmark_outputs, batch_size, 10).reshape(-1, 5, 2)
            masks = torch.cat(selected_landmark_masks)
            selected_predictions = predictions[positive_mask][masks]
            selected_targets = torch.cat(landmark_targets)[masks]
            loss_landmark = F.smooth_l1_loss(
                selected_predictions, selected_targets, reduction="sum",
            ) / max(selected_predictions.numel(), 1)
            losses["landmark"] = loss_landmark * self.weights.landmark
            losses["total"] = losses["total"] + losses["landmark"]
        return losses


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


def _bbox_decode(priors: torch.Tensor, regressions: torch.Tensor) -> torch.Tensor:
    centers = regressions[..., :2] * priors[..., 2:] + priors[..., :2]
    sizes = regressions[..., 2:].exp() * priors[..., 2:]
    return torch.cat([centers - sizes / 2, centers + sizes / 2], dim=-1)


def _bbox_overlaps(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    intersection_size = (bottom_right - top_left).clamp(min=0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    return intersection / (area1[:, None] + area2 - intersection + 1e-6)


def _eiou_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    smooth_point: float = 0.1,
) -> torch.Tensor:
    px1, py1, px2, py2 = predicted.unbind(dim=1)
    tx1, ty1, tx2, ty2 = target.unbind(dim=1)
    enclosure_x1 = torch.minimum(px1, tx1)
    enclosure_y1 = torch.minimum(py1, ty1)
    intersection_x1 = torch.maximum(px1, tx1)
    intersection_y1 = torch.maximum(py1, ty1)
    intersection_x2 = torch.minimum(px2, tx2)
    intersection_y2 = torch.minimum(py2, ty2)
    minimum_x = torch.minimum(intersection_x1, intersection_x2)
    minimum_y = torch.minimum(intersection_y1, intersection_y2)
    maximum_x = torch.maximum(intersection_x1, intersection_x2)
    maximum_y = torch.maximum(intersection_y1, intersection_y2)
    intersection = (
        (intersection_x2 - enclosure_x1) * (intersection_y2 - enclosure_y1)
        + (minimum_x - enclosure_x1) * (minimum_y - enclosure_y1)
        - (intersection_x1 - enclosure_x1) * (maximum_y - enclosure_y1)
        - (maximum_x - enclosure_x1) * (intersection_y1 - enclosure_y1)
    )
    union = (
        (px2 - px1) * (py2 - py1)
        + (tx2 - tx1) * (ty2 - ty1)
        - intersection
        + 1e-7
    )
    distance = 1 - intersection / union
    smooth = (distance < smooth_point).detach().to(distance.dtype)
    return 0.5 * smooth * distance.square() / smooth_point + (1 - smooth) * (
        distance - 0.5 * smooth_point
    )
