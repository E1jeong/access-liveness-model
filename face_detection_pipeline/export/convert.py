"""Export YuNet-N through the existing ONNX/onnx2tf/TFLite toolchain.

Input: NHWC RGB float32 pixels in [0, 255], centered black letterbox.
Outputs: logits [1, N, 2], boxes [1, N, 4], optional landmarks [1, N, 10], stride 8/16/32,
row-major within each level. Separate tensors preserve independent INT8 scales.
Decode, sigmoid, NMS and source-coordinate restoration stay outside the graph.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from common.utils import subsample_split_items
from face_detection_pipeline.data.dataset import FaceDetectionDataset, LANDMARK_NAMES, _validate_target_size
from face_detection_pipeline.evaluation.detection import (
    calculate_detection_metrics, calculate_landmark_metrics, decode_detections,
)
from face_detection_pipeline.models.yunet import YuNetLandmarkOutputs, YuNetOutputs, build_yunet_n
from face_detection_pipeline.training.train import UPSTREAM_COMMIT


class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        features = self.model.backbone(images)
        for index in range(len(features) - 1, 0, -1):
            features[index] = self.model.neck.lateral_convs[index](features[index])
            # Fixed multiples-of-32 inputs make every pyramid level exactly 2x.
            # Export constant scales instead of shape-derived Resize sizes.
            features[index - 1] = features[index - 1] + torch.nn.functional.interpolate(
                features[index], scale_factor=2.0, mode="nearest")
        features[0] = self.model.neck.lateral_convs[0](features[0])
        outputs = self.model.head(features)
        logits = [torch.cat([cls, obj], dim=1).permute(0, 2, 3, 1).reshape(1, -1, 2)
                  for cls, obj in zip(outputs.cls_scores, outputs.objectnesses)]
        boxes = [box.permute(0, 2, 3, 1).reshape(1, -1, 4) for box in outputs.bbox_regressions]
        parts = (torch.cat(logits, dim=1), torch.cat(boxes, dim=1))
        if self.model.landmarks:
            points = [point.permute(0, 2, 3, 1).reshape(1, -1, 10)
                      for point in outputs.landmark_regressions]
            parts += (torch.cat(points, dim=1),)
        return parts


def unpack_predictions(predictions, image_size):
    width, height = image_size
    count = sum((width // stride) * (height // stride) for stride in (8, 16, 32))
    if predictions.ndim != 3 or predictions.shape[:2] != (1, count) or predictions.shape[-1] not in (6, 16):
        raise ValueError(f"Unexpected detector output shape: {predictions.shape}")
    channels = predictions.shape[-1]
    groups = [[], [], []] if channels == 6 else [[], [], [], []]
    offset = 0
    for stride in (8, 16, 32):
        h, w = height // stride, width // stride
        level = predictions[:, offset:offset + h * w].reshape(1, h, w, channels).permute(0, 3, 1, 2)
        parts = (level[:, :1], level[:, 1:2], level[:, 2:6])
        if channels == 16:
            parts += (level[:, 6:],)
        for group, values in zip(groups, parts):
            group.append(values)
        offset += h * w
    return YuNetOutputs(*groups) if channels == 6 else YuNetLandmarkOutputs(*groups)


def load_checkpoint(path):
    contents = path.read_bytes()
    checkpoint = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
    if checkpoint.get("model") not in ("yunet_n_box_only", "yunet_n_landmarks"):
        raise ValueError("Expected a YuNet-N training checkpoint")
    landmarks = checkpoint["model"] == "yunet_n_landmarks"
    if landmarks and (checkpoint.get("landmark_names") != list(LANDMARK_NAMES)
                      or checkpoint.get("landmark_encoding") != "grid_relative_stride_units_xy"):
        raise ValueError("Invalid checkpoint landmark contract")
    if checkpoint.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Checkpoint YuNet source revision does not match")
    image_size = tuple(checkpoint.get("input_size", ()))
    _validate_target_size(image_size)
    if not isinstance(checkpoint.get("run_id"), str) or not isinstance(checkpoint.get("epoch"), int):
        raise ValueError("Checkpoint must record its training run_id and epoch")
    model = build_yunet_n(landmarks=landmarks).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return ExportModel(model).eval(), checkpoint, hashlib.sha256(contents).hexdigest()


def validate_output_dir(path):
    if not path.resolve().is_relative_to(Path("model").resolve()):
        raise ValueError("Export output must be under gitignored model/")
    if path.exists():
        raise FileExistsError(f"Export directory already exists: {path}")


def validate_tflite_contract(inputs, outputs, image_size, int8, *, landmarks=False):
    width, height = image_size
    count = sum((width // s) * (height // s) for s in (8, 16, 32))
    dtype = np.int8 if int8 else np.float32
    if len(outputs) != (3 if landmarks else 2):
        raise ValueError("Expected separate logits, boxes and optional landmark outputs")
    outputs = sorted(outputs, key=lambda info: int(info["shape"][-1]))
    contracts = [(inputs, [1, height, width, 3]),
                 (outputs[:1], [1, count, 2]), (outputs[1:2], [1, count, 4])]
    if landmarks:
        contracts.append((outputs[2:], [1, count, 10]))
    for details, shape in contracts:
        if len(details) != 1 or list(details[0]["shape"]) != shape or details[0]["dtype"] != dtype:
            raise ValueError(f"Invalid detector tensor contract; expected {shape}, {dtype}: {details}")
        if int8 and details[0].get("quantization", (0, 0))[0] <= 0:
            raise ValueError("INT8 tensor must have a positive quantization scale")


def _select_indices(dataset, count, seed):
    items = [(index, None, path.relative_to(dataset.split_dir).parts[0])
             for index, path in enumerate(dataset.records)]
    return [item[0] for item in subsample_split_items(items, count, seed)]


def _image_array(image):
    return image.permute(1, 2, 0).unsqueeze(0).numpy().astype(np.float32)


def _interpreter(model_bytes, image_size, int8, *, landmarks=False):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_content=model_bytes, num_threads=2)
    interpreter.allocate_tensors()
    validate_tflite_contract(interpreter.get_input_details(), interpreter.get_output_details(), image_size, int8,
                             landmarks=landmarks)
    custom = [op["op_name"] for op in interpreter._get_ops_details()
              if op["op_name"].startswith("Flex") or op["op_name"] == "CUSTOM"]
    if custom:
        raise ValueError(f"Detector requires unsupported custom/Flex operators: {custom}")
    return interpreter


def _predict(interpreter, image):
    input_info = interpreter.get_input_details()[0]
    values = _image_array(image)
    if input_info["dtype"] == np.int8:
        scale, zero = input_info["quantization"]
        values = np.clip(np.rint(values / scale + zero), -128, 127).astype(np.int8)
    interpreter.set_tensor(input_info["index"], values)
    interpreter.invoke()
    parts = []
    for output_info in sorted(interpreter.get_output_details(), key=lambda info: int(info["shape"][-1])):
        result = interpreter.get_tensor(output_info["index"])
        if output_info["dtype"] == np.int8:
            scale, zero = output_info["quantization"]
            result = (result.astype(np.float32) - zero) * scale
        parts.append(result)
    result = np.concatenate(parts, axis=-1)
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite TFLite prediction")
    return result


def _verify(model, interpreter, dataset, indices, image_size, *, require_parity):
    reference_detections, converted_detections, targets = [], [], []
    max_error = 0.0
    golden_samples = []
    landmark_targets, landmark_masks = [], []
    max_landmark_error = 0.0
    max_source_landmark_error = 0.0
    for index in indices:
        sample = dataset[index]
        image, boxes = sample[:2]
        if model.model.landmarks:
            landmark_targets.append(sample[2])
            landmark_masks.append(sample[3])
        with torch.inference_mode():
            reference = torch.cat(model(image.unsqueeze(0)), dim=-1).numpy()
        actual = _predict(interpreter, image)
        max_error = max(max_error, float(np.max(np.abs(reference - actual))))
        if require_parity:
            np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-3)
        decoded = [decode_detections(unpack_predictions(torch.from_numpy(predictions), image_size))[0]
                   for predictions in (reference, actual)]
        if require_parity:
            np.testing.assert_allclose(decoded[1].boxes.numpy(), decoded[0].boxes.numpy(), rtol=1e-3, atol=0.1)
            np.testing.assert_allclose(decoded[1].scores.numpy(), decoded[0].scores.numpy(), rtol=1e-3, atol=1e-3)
        if model.model.landmarks:
            # Compare every decoded point, including candidates removed by NMS.
            ref_offsets = reference[..., 6:].reshape(1, -1, 5, 2)
            actual_offsets = actual[..., 6:].reshape(1, -1, 5, 2)
            stride_values = np.concatenate([
                np.full((image_size[1] // stride) * (image_size[0] // stride), stride)
                for stride in (8, 16, 32)
            ])
            error = np.abs(actual_offsets - ref_offsets) * stride_values[None, :, None, None]
            max_landmark_error = max(max_landmark_error, float(error.max()))
            rgb = json.loads(dataset.records[index].with_name("meta.json").read_text())["rgb"]
            scale = min(image_size[0] / rgb["width"], image_size[1] / rgb["height"])
            source_error = float(error.max()) / scale
            max_source_landmark_error = max(max_source_landmark_error, source_error)
            if require_parity:
                assert float(error.max()) <= 0.1, "Decoded landmark parity exceeds 0.1 input pixel"
                assert source_error <= 0.1, "Decoded landmark parity exceeds 0.1 original RGB pixel"
                np.testing.assert_allclose(decoded[1].landmarks.numpy(), decoded[0].landmarks.numpy(),
                                           rtol=1e-3, atol=0.1)
        reference_detections.append(decoded[0])
        converted_detections.append(decoded[1])
        targets.append(boxes)
        path = dataset.records[index]
        golden_samples.append({"path": str(path.relative_to(dataset.split_dir)),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    result = {
        "samples": len(indices), "max_abs_raw_error": max_error,
        "float_parity_passed": True if require_parity else None,
        "pytorch": calculate_detection_metrics(reference_detections, targets),
        "tflite": calculate_detection_metrics(converted_detections, targets),
        "golden_samples": golden_samples,
    }
    if model.model.landmarks:
        result["max_abs_decoded_landmark_error_pixels"] = max_landmark_error
        result["max_abs_source_landmark_error_pixels"] = max_source_landmark_error
        result["landmarks"] = {
            "coordinate_system": "letterboxed_input_pixels",
            "reference": "selected pseudo-label points; not reviewed ground truth",
            "pytorch": calculate_landmark_metrics(reference_detections, targets, landmark_targets, landmark_masks),
            "tflite": calculate_landmark_metrics(converted_detections, targets, landmark_targets, landmark_masks),
        }
    return result


def convert(args):
    validate_output_dir(args.output_dir)
    print(f"[load checkpoint] {args.checkpoint}", flush=True)
    model, checkpoint, checkpoint_hash = load_checkpoint(args.checkpoint)
    image_size = tuple(checkpoint["input_size"])
    landmarks = model.model.landmarks
    validation = FaceDetectionDataset(args.data_dir / "validation", image_size=image_size, landmarks=landmarks)
    indices = _select_indices(validation, args.validation_samples, args.seed)
    calibration = None
    if args.int8:
        calibration = FaceDetectionDataset(args.data_dir / "train", image_size=image_size)
        calibration_indices = _select_indices(calibration, args.calibration_samples, args.seed)
    args.output_dir.mkdir(parents=True)

    # Heavy conversion imports are deliberately lazy; invalid checkpoints fail first.
    import onnx2tf
    import tensorflow as tf

    width, height = image_size
    with tempfile.TemporaryDirectory(prefix="convert-", dir=args.output_dir) as temporary:
        onnx_path = Path(temporary) / "yunet.onnx"
        saved_model = Path(temporary) / "saved_model"
        example = validation[indices[0]][0].unsqueeze(0)
        output_names = ["logits", "box_regressions"] + (["landmark_regressions"] if landmarks else [])
        torch.onnx.export(model, example, str(onnx_path), input_names=["rgb"],
                          output_names=output_names, opset_version=17, dynamo=False)
        print("[export] ONNX -> SavedModel", flush=True)
        sample_path = Path(temporary) / "golden.npy"
        golden = _image_array(example[0])
        np.save(sample_path, golden)
        # onnx2tf 1.29 downloads sample images even with custom input supplied.
        # Supply our own golden RGB instead; conversion must stay offline.
        with patch("onnx2tf.onnx2tf.download_test_image_data", return_value=golden):
            onnx2tf.convert(input_onnx_file_path=str(onnx_path), output_folder_path=str(saved_model),
                            custom_input_op_name_np_data_path=[["rgb", str(sample_path)]],
                            # At stride 32, 320x320 has a 10x10x10 landmark map;
                            # strict validation disambiguates its equal-sized axes.
                            disable_strict_mode=not landmarks, non_verbose=True)
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        float_bytes = converter.convert()
        interpreter = _interpreter(float_bytes, image_size, False, landmarks=landmarks)
        report = {"float": _verify(model, interpreter, validation, indices, image_size, require_parity=True)}
        float_path = args.output_dir / "yunet_n_float.tflite"
        float_path.write_bytes(float_bytes)
        print(f"[float parity passed] {float_path}", flush=True)
        artifacts = {"float": _artifact(float_path, interpreter)}
        if args.int8:
            def representative():
                for index in calibration_indices:
                    yield [_image_array(calibration[index][0])]

            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            int8_bytes = converter.convert()
            interpreter = _interpreter(int8_bytes, image_size, True, landmarks=landmarks)
            report["int8"] = _verify(model, interpreter, validation, indices, image_size, require_parity=False)
            int8_path = args.output_dir / "yunet_n_int8.tflite"
            int8_path.write_bytes(int8_bytes)
            artifacts["int8"] = _artifact(int8_path, interpreter)
            report["calibration"] = {"split": "train", "samples": len(calibration_indices),
                                     "paths": [str(calibration.records[i].relative_to(calibration.split_dir))
                                               for i in calibration_indices]}
            print(f"[int8 saved; inspect validation metrics before deployment] {int8_path}", flush=True)
    snapshot_path = Path("source-manifest.json")
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.is_file() else None
    metadata = {
        "format_version": 2 if landmarks else 1, "model": checkpoint["model"], "upstream_commit": UPSTREAM_COMMIT,
        "run_id": checkpoint["run_id"], "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": checkpoint_hash, "machine": platform.node(),
        "export_code_revision": (snapshot["base_revision"] if snapshot else
                                 subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()),
        "export_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "export_worktree_dirty": (True if snapshot else
                                  bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())),
        **({"export_source_snapshot": snapshot} if snapshot else {}),
        "created_utc": datetime.now(timezone.utc).isoformat(), "validation_split": "validation",
        "toolchain": {name: version(name) for name in ("torch", "tensorflow", "onnx", "onnx2tf")},
        "input": {"layout": "NHWC", "shape": [1, height, width, 3], "color": "RGB",
                  "pixel_range": [0, 255], "resize": "bilinear centered letterbox, black padding"},
        "output": {"layout": "[batch, candidates, channels]", "strides": [8, 16, 32],
                   "order": "stride ascending, then row-major y/x",
                   "logits_channels": ["class_logit", "objectness_logit"],
                   "box_channels": ["dx", "dy", "log_width", "log_height"],
                   "binding": "Identify logits by last dimension 2, boxes by last dimension 4; use artifact tensor indices"},
        "postprocess": {"score": "sigmoid(class_logit) * sigmoid(objectness_logit)",
                        "center": "(grid_xy + delta_xy) * stride", "size": "exp(clip(log_wh, -16, 16)) * stride",
                        "score_threshold": 0.5, "nms_threshold": 0.45, "iou_threshold": 0.5,
                        "max_detections": 100, "restore": "subtract padding, divide by letterbox scale, clip"},
        "npu_verified": False, "artifacts": artifacts, "validation": report,
    }
    if landmarks:
        metadata["output"]["landmark_names"] = list(LANDMARK_NAMES)
        metadata["output"]["landmark_channels"] = [f"{name}_{axis}" for name in LANDMARK_NAMES for axis in ("x", "y")]
        metadata["output"]["binding"] = "Last dimensions: logits 2, boxes 4, landmarks 10; use artifact tensor indices"
        metadata["postprocess"]["landmarks"] = "(grid_xy + offset_xy) * stride; same candidate/NMS indices as boxes"
        metadata["postprocess"]["landmark_restore"] = "subtract padding, divide by letterbox scale, clip to original RGB bounds"
        metadata["postprocess"]["landmark_visibility"] = "not predicted; training masks are supervision only"
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({kind: {k: v for k, v in result.items() if k != "golden_samples"}
                      for kind, result in report.items() if kind != "calibration"}, indent=2), flush=True)
    return args.output_dir


def _artifact(path, interpreter):
    def tensor(info):
        return {"index": int(info["index"]), "name": info["name"], "shape": info["shape"].tolist(),
                "dtype": np.dtype(info["dtype"]).name, "quantization": list(info["quantization"])}
    return {"file": path.name, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "input": tensor(interpreter.get_input_details()[0]),
            "outputs": [tensor(info) for info in interpreter.get_output_details()]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory under model/")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--int8", action="store_true", help="Also export full INT8 after float parity passes")
    parser.add_argument("--calibration-samples", type=int, default=200)
    parser.add_argument("--validation-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.calibration_samples <= 0 or args.validation_samples <= 0:
        raise ValueError("calibration-samples and validation-samples must be positive")
    return args


if __name__ == "__main__":
    torch.set_num_threads(2)
    convert(parse_args())
