from pathlib import Path
import pytest
from face_recognition_pipeline.export.export_npu import export_npu_model


pytestmark = pytest.mark.tensorflow


def test_export_npu_model_refuses_overwriting_without_force(tmp_path):
    dummy_out = tmp_path / "output.tflite"
    dummy_out.write_bytes(b"existing_model")

    with pytest.raises(FileExistsError, match="--force was not specified"):
        export_npu_model(
            weights_path="non_existent.h5",
            output_path=str(dummy_out),
            dataset_root=str(tmp_path),
            force=False,
        )
