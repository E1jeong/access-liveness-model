"""Train the box-only YuNet-N detector on the fixed internal split."""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from face_detection_pipeline.data.dataset import (
    FaceDetectionDataset,
    collate_face_detection_batch,
)
from face_detection_pipeline.evaluation.detection import (
    Detection,
    calculate_detection_metrics,
    decode_detections,
)
from face_detection_pipeline.models.yunet import build_yunet_n
from face_detection_pipeline.training.loss import YuNetCriterion

UPSTREAM_COMMIT = "dca340aa082c71081a68d17db8e58b33a58a914b"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train box-only YuNet-N")
    parser.add_argument("--data-dir", default="dataset/raw")
    parser.add_argument("--output-dir", default="model/face_detection")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-iters", type=int, default=1500)
    parser.add_argument("--warmup-ratio", type=float, default=0.001)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-validation-samples", type=int)
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def train(args: argparse.Namespace) -> Path:
    _validate_args(args)
    _seed_everything(args.seed)
    device = _resolve_device(args.device)
    image_size = (args.image_width, args.image_height)
    data_dir = Path(args.data_dir)
    train_dataset = FaceDetectionDataset(
        data_dir / "train",
        image_size=image_size,
        limit_samples=args.limit_train_samples,
    )
    validation_dataset = FaceDetectionDataset(
        data_dir / "validation",
        image_size=image_size,
        limit_samples=args.limit_validation_samples,
    )
    train_loader = _build_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        device=device,
    )
    validation_loader = _build_loader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        device=device,
    )

    model = build_yunet_n().to(device)
    criterion = YuNetCriterion()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    warmup = _LinearWarmup(
        optimizer,
        iterations=args.warmup_iters,
        start_ratio=args.warmup_ratio,
    )
    if args.resume is None:
        run_id = args.run_id or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ_yunet_n"
        )
        run_dir = Path(args.output_dir) / run_id
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        start_epoch = 1
        history = []
        best_validation_f1 = -1.0
        best_validation_loss = float("inf")
    else:
        checkpoint = _load_checkpoint(args.resume, model, optimizer, warmup, device)
        if checkpoint.get("input_size") != [args.image_width, args.image_height]:
            raise ValueError(
                f"Checkpoint input size {checkpoint.get('input_size')} does not match "
                f"requested {[args.image_width, args.image_height]}"
            )
        run_id = str(checkpoint["run_id"])
        if args.run_id is not None and args.run_id != run_id:
            raise ValueError(
                f"run-id {args.run_id!r} does not match checkpoint run {run_id!r}"
            )
        run_dir = args.resume.parent
        start_epoch = int(checkpoint["epoch"]) + 1
        if start_epoch > args.epochs:
            raise ValueError(
                f"Checkpoint is already at epoch {start_epoch - 1}; "
                f"requested epochs={args.epochs}"
            )
        history = _load_history(
            run_dir / f"metrics_yunet_n_{args.image_width}x{args.image_height}.json"
        )
        best_validation_f1 = float(checkpoint.get("best_validation_f1", -1.0))
        best_validation_loss = float(
            checkpoint.get(
                "best_validation_loss",
                checkpoint["validation_metrics"]["total"],
            )
        )
    best_path = run_dir / f"best_yunet_n_{args.image_width}x{args.image_height}.pth"
    latest_path = run_dir / f"latest_yunet_n_{args.image_width}x{args.image_height}.pth"
    metrics_path = run_dir / f"metrics_yunet_n_{args.image_width}x{args.image_height}.json"

    print(
        f"device={device} train={len(train_dataset)} validation={len(validation_dataset)} "
        f"input={args.image_width}x{args.image_height}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            warmup=warmup,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            criterion,
            optimizer=None,
            device=device,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            iou_threshold=args.iou_threshold,
            max_detections=args.max_detections,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        validation_f1 = validation_metrics["f1"]
        is_best = validation_f1 > best_validation_f1 or (
            validation_f1 == best_validation_f1
            and validation_metrics["total"] < best_validation_loss
        )
        if is_best:
            best_validation_f1 = validation_f1
            best_validation_loss = validation_metrics["total"]
        checkpoint = _checkpoint(
            model,
            optimizer,
            args,
            run_id,
            epoch,
            validation_metrics,
            warmup,
            best_validation_f1,
            best_validation_loss,
        )
        torch.save(checkpoint, latest_path)
        if is_best:
            torch.save(checkpoint, best_path)
        metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"epoch={epoch}/{args.epochs} "
            f"train_loss={train_metrics['total']:.6f} "
            f"validation_loss={validation_metrics['total']:.6f} "
            f"precision={validation_metrics['precision']:.4f} "
            f"recall={validation_metrics['recall']:.4f} "
            f"f1={validation_metrics['f1']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.8f}"
        )

    print(f"best_checkpoint={best_path}")
    return best_path


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: YuNetCriterion,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    warmup: "_LinearWarmup | None" = None,
    score_threshold: float | None = None,
    nms_threshold: float = 0.45,
    iou_threshold: float = 0.5,
    max_detections: int = 100,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in ("total", "classification", "objectness", "bbox")}
    sample_count = 0
    all_detections: list[Detection] = []
    all_ground_truth: list[torch.Tensor] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, boxes in loader:
            if warmup is not None:
                warmup.step()
            images = images.to(device, non_blocking=True)
            boxes = [box.to(device, non_blocking=True) for box in boxes]
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            losses = criterion(outputs, boxes)
            non_finite = [name for name, value in losses.items() if not torch.isfinite(value)]
            if non_finite:
                names = ", ".join(non_finite)
                raise FloatingPointError(f"Non-finite loss detected: {names}")
            if score_threshold is not None:
                batch_detections = decode_detections(
                    outputs,
                    score_threshold=score_threshold,
                    nms_threshold=nms_threshold,
                    max_detections=max_detections,
                )
                all_detections.extend(
                    Detection(item.boxes.detach().cpu(), item.scores.detach().cpu())
                    for item in batch_detections
                )
                all_ground_truth.extend(box.detach().cpu() for box in boxes)
            if training:
                losses["total"].backward()
                optimizer.step()
            batch_size = images.shape[0]
            sample_count += batch_size
            for name, value in losses.items():
                totals[name] += float(value.detach()) * batch_size
    if sample_count == 0:
        raise ValueError("Cannot train or validate with an empty dataset")
    metrics = {name: value / sample_count for name, value in totals.items()}
    if score_threshold is not None:
        metrics.update(
            calculate_detection_metrics(
                all_detections,
                all_ground_truth,
                iou_threshold=iou_threshold,
            )
        )
    return metrics


def _build_loader(
    dataset: FaceDetectionDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_face_detection_batch,
    )


def _checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    run_id: str,
    epoch: int,
    validation_metrics: dict[str, float],
    warmup: "_LinearWarmup",
    best_validation_f1: float,
    best_validation_loss: float,
) -> dict:
    return {
        "format_version": 1,
        "model": "yunet_n_box_only",
        "upstream_commit": UPSTREAM_COMMIT,
        "run_id": run_id,
        "input_size": [args.image_width, args.image_height],
        "training_config": _serializable_training_config(args),
        "epoch": epoch,
        "validation_metrics": validation_metrics,
        "best_validation_f1": best_validation_f1,
        "best_validation_loss": best_validation_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "warmup_state": warmup.state_dict(),
    }


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _validate_args(args: argparse.Namespace) -> None:
    if args.image_width <= 0 or args.image_width % 32:
        raise ValueError("image-width must be a positive multiple of 32")
    if args.image_height <= 0 or args.image_height % 32:
        raise ValueError("image-height must be a positive multiple of 32")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if args.learning_rate <= 0 or args.num_workers < 0:
        raise ValueError("learning-rate must be positive and num-workers non-negative")
    if args.warmup_iters < 0 or not 0 < args.warmup_ratio <= 1:
        raise ValueError("warmup-iters must be non-negative and warmup-ratio in (0, 1]")
    if not 0 <= args.score_threshold <= 1 or not 0 <= args.nms_threshold <= 1:
        raise ValueError("score-threshold and nms-threshold must be in [0, 1]")
    if not 0 < args.iou_threshold <= 1 or args.max_detections <= 0:
        raise ValueError("iou-threshold must be in (0, 1] and max-detections positive")
    if args.run_id is not None and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*",
        args.run_id,
    ):
        raise ValueError("run-id must contain only letters, numbers, dot, dash, or underscore")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _LinearWarmup:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        iterations: int,
        start_ratio: float,
    ):
        self.optimizer = optimizer
        self.iterations = iterations
        self.start_ratio = start_ratio
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.completed_iterations = 0

    def step(self) -> None:
        factor = 1.0
        if self.iterations and self.completed_iterations < self.iterations:
            progress = self.completed_iterations / self.iterations
            factor = self.start_ratio + (1.0 - self.start_ratio) * progress
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor
        self.completed_iterations += 1

    def state_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "start_ratio": self.start_ratio,
            "base_lrs": self.base_lrs,
            "completed_iterations": self.completed_iterations,
        }

    def load_state_dict(self, state: dict) -> None:
        self.iterations = int(state["iterations"])
        self.start_ratio = float(state["start_ratio"])
        self.base_lrs = [float(value) for value in state["base_lrs"]]
        self.completed_iterations = int(state["completed_iterations"])


def _load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    warmup: _LinearWarmup,
    device: torch.device,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model") != "yunet_n_box_only":
        raise ValueError(f"Unsupported checkpoint model: {checkpoint.get('model')!r}")
    if checkpoint.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Checkpoint YuNet source revision does not match this implementation")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    warmup.load_state_dict(checkpoint["warmup_state"])
    return checkpoint


def _load_history(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Metrics history does not exist: {path}")
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history, list):
        raise ValueError(f"Metrics history must be a JSON list: {path}")
    return history


def _serializable_training_config(args: argparse.Namespace) -> dict:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


if __name__ == "__main__":
    train(parse_args())
