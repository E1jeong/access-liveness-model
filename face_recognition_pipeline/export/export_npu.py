"""Export NPU-compliant INT8 TFLite model from MobileNetV1 face recognition checkpoint."""

import argparse
import os
from pathlib import Path

import tensorflow as tf

from face_recognition_pipeline.data.calibration import (
    CALIBRATION_SAMPLES,
    collect_live_face_paths,
    load_normalized_face_image,
    select_stratified_face_paths,
)
from face_recognition_pipeline.export.verifier import verify_int8_operator_graph
from face_recognition_pipeline.models.mobilenet_v1 import build_npu_mobilenet_model


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Export NPU-compliant INT8 TFLite face recognition model."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(repo_root / "model" / "recognition" / "mobilenet_emb256.h5"),
        help="Path to pre-trained Keras .h5 weights file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(
            repo_root / "model" / "recognition" / "tflite" / "mobilenet_emore_npu_int8.tflite"
        ),
        help="Path to output .tflite file.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(repo_root / "dataset" / "raw"),
        help="Root path containing fixed dataset splits ({train,validation}/live).",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=CALIBRATION_SAMPLES,
        help=f"Number of calibration samples (default: {CALIBRATION_SAMPLES}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for calibration sample selection.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="기존 산출물 덮어쓰기 허용 (Run & Data Protection 정책 준수)",
    )
    return parser.parse_args()


def export_npu_model(
    weights_path: str,
    output_path: str,
    dataset_root: str,
    calib_samples: int = CALIBRATION_SAMPLES,
    seed: int = 42,
    force: bool = False,
) -> None:
    """Build, calibrate, quantize, verify and export the NPU recognition model."""
    out_file = Path(output_path)
    if out_file.exists() and not force:
        raise FileExistsError(
            f"Artifact already exists and --force was not specified: {out_file} "
            "(Run & Data Protection policy)."
        )

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Recognition weights file not found: {weights_path}")

    print(f"Building NPU MobileNetV1 model from weights: {weights_path}")
    model = build_npu_mobilenet_model(weights_path=weights_path)
    print("Weights loaded successfully into NPU-compliant model.")

    print(f"Sampling calibration images from {dataset_root}...")
    samples_by_subject = collect_live_face_paths(dataset_root)
    calibration_paths, selected_by_subject = select_stratified_face_paths(
        samples_by_subject,
        sample_count=calib_samples,
        seed=seed,
    )
    print(
        f"Selected {len(calibration_paths)} calibration images across "
        f"{len(selected_by_subject)} live subjects."
    )

    def rep_gen():
        for image_path in calibration_paths:
            yield [load_normalized_face_image(image_path)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("Quantizing model to INT8...")
    tflite_int8 = converter.convert()

    print("Verifying INT8 NPU operator graph contract...")
    verify_int8_operator_graph(tflite_int8)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f_out:
        f_out.write(tflite_int8)

    print(f"Exported NPU INT8 model ({len(tflite_int8) / (1024 * 1024):.2f} MB) to {out_file}")


def main():
    args = parse_args()
    export_npu_model(
        weights_path=args.weights,
        output_path=args.output,
        dataset_root=args.dataset_root,
        calib_samples=args.calib_samples,
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
