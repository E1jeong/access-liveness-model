import os
import tempfile
import json
import pytest
import tensorflow as tf

from classes import CLASS_NAMES
from pytorch_pipeline.convert_to_tflite import convert_pytorch_to_tflite

def test_pytorch_mct_tflite_export_and_manifest():
    with tempfile.TemporaryDirectory() as temp_dir:
        output_prefix = os.path.join(temp_dir, "test_crop_ir")
        
        convert_pytorch_to_tflite(
            pth_path="non_existent.pth",
            output_prefix=output_prefix,
            model_type="crop_ir",
            num_classes=len(CLASS_NAMES),
            calib_samples=5,
            dataset_dir="dataset/raw/train"
        )
        
        # Verify generated files
        float_tflite = f"{output_prefix}_float.tflite"
        float_manifest = f"{output_prefix}_float_manifest.json"
        int8_tflite = f"{output_prefix}_int8.tflite"
        int8_manifest = f"{output_prefix}_int8_manifest.json"
        npu_int8_tflite = f"{output_prefix}_npu_int8.tflite"
        npu_int8_manifest = f"{output_prefix}_npu_int8_manifest.json"
        
        assert os.path.exists(float_tflite)
        assert os.path.exists(float_manifest)
        assert os.path.exists(int8_tflite)
        assert os.path.exists(int8_manifest)
        assert os.path.exists(npu_int8_tflite)
        assert os.path.exists(npu_int8_manifest)
        
        # Verify NPU INT8 Interpreter tensor signatures
        interpreter = tf.lite.Interpreter(model_path=npu_int8_tflite)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        assert len(input_details) == 1
        assert list(input_details[0]["shape"]) == [1, 224, 224, 1]
        assert input_details[0]["dtype"].__name__ == "int8"
        assert "b_crop_ir" in input_details[0]["name"]
        
        assert len(output_details) == 1
        assert list(output_details[0]["shape"]) == [1, 10]
        assert output_details[0]["dtype"].__name__ == "int8"
        
        # Verify sidecar manifest content
        with open(npu_int8_manifest, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            
        assert manifest_data["model_type"] == "crop_ir"
        assert manifest_data["delegate"] == "nnapi"
        assert manifest_data["class_order"] == CLASS_NAMES
        assert manifest_data["crop_margin_ratio"] == 0.10
        assert manifest_data["inputs"][0]["input_kind"] == "ir"
        assert manifest_data["outputs"][0]["output_is_logits"] is True
