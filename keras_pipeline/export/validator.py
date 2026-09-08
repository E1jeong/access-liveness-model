"""Inspect exported TFLite artifacts and write their Android sidecar contracts."""
import json
import os

import tensorflow as tf

from common.classes import CLASS_NAMES
from keras_pipeline.contracts.model_signature import validate_tflite_model_signature


def inspect_tflite(path, model_type):
    """Validate a saved TFLite signature and print its tensor details."""
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    validate_tflite_model_signature(input_details, output_details, model_type)
    print("[tflite tensors]")
    for idx, detail in enumerate(input_details):
        print(
            f" input {idx}: name={detail['name']} shape={detail['shape'].tolist()} "
            f"dtype={detail['dtype']} quant={detail['quantization']}"
        )
    for idx, detail in enumerate(output_details):
        print(
            f" output {idx}: name={detail['name']} shape={detail['shape'].tolist()} "
            f"dtype={detail['dtype']} quant={detail['quantization']}"
        )


def write_tflite_sidecar_manifest(tflite_path, model_type):
    """Write the Android input/output contract next to a TFLite artifact."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    is_npu_int8 = "npu_int8" in os.path.basename(tflite_path)

    inputs_info = []
    for idx, detail in enumerate(input_details):
        name = detail['name']
        shape = detail['shape'].tolist()
        dtype = detail['dtype'].__name__
        channels = shape[-1]

        input_kind = "unknown"
        if model_type == "crop_rgb":
            input_kind = "rgb"
        elif model_type == "crop_ir":
            input_kind = "ir"
        elif model_type == "dual":
            input_kind = "rgb" if channels == 3 else "ir"

        scale, zero_point = detail['quantization']
        quant = None
        if scale != 0.0 or zero_point != 0:
            quant = {
                "scale": float(scale),
                "zero_point": int(zero_point)
            }

        if channels == 3:
            if is_npu_int8:
                norm = {
                    "mean": [0.5, 0.5, 0.5],
                    "std": [0.5, 0.5, 0.5],
                    "range": "[-1, 1]"
                }
            else:
                norm = {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "range": "imagenet"
                }
        else:
            norm = {
                "mean": [0.5],
                "std": [0.5],
                "range": "[-1, 1]"
            }

        inputs_info.append({
            "name": name,
            "index": idx,
            "shape": shape,
            "dtype": dtype,
            "input_kind": input_kind,
            "quantization": quant,
            "normalization": norm
        })

    outputs_info = []
    for idx, detail in enumerate(output_details):
        name = detail['name']
        shape = detail['shape'].tolist()
        dtype = detail['dtype'].__name__

        scale, zero_point = detail['quantization']
        quant = None
        if scale != 0.0 or zero_point != 0:
            quant = {
                "scale": float(scale),
                "zero_point": int(zero_point)
            }

        outputs_info.append({
            "name": name,
            "index": idx,
            "shape": shape,
            "dtype": dtype,
            "quantization": quant,
            "output_is_logits": True
        })

    manifest = {
        "model_type": model_type,
        "file_name": os.path.basename(tflite_path),
        "delegate": "nnapi" if is_npu_int8 else "cpu",
        "inputs": inputs_info,
        "outputs": outputs_info,
        "class_order": CLASS_NAMES,
        "crop_margin_ratio": 0.10
    }

    manifest_path = tflite_path.replace(".tflite", "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[tflite sidecar manifest saved] {manifest_path}")
