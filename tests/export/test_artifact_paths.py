"""keras_pipeline.artifact_paths의 이름 규칙과 덮어쓰기 방지를 검증한다."""
import os

import pytest

from keras_pipeline.artifact_paths import (
    DEFAULT_OUTPUT_DIR,
    TFLITE_VARIANTS,
    keras_checkpoint_name,
    keras_checkpoint_path,
    tflite_name,
    tflite_path,
    tflite_paths,
    sidecar_manifest_path,
    calibration_manifest_name,
    calibration_manifest_path,
    learning_curves_name,
    learning_curves_path,
    metadata_name,
    metadata_path,
    check_no_overwrite,
)


# ── 체크포인트 이름 ───────────────────────────────────────────────

class TestKerasCheckpoint:
    def test_dual(self):
        assert keras_checkpoint_name("dual") == "best_model_fixed.keras"

    def test_crop_rgb(self):
        assert keras_checkpoint_name("crop_rgb") == "best_crop_rgb_fixed.keras"

    def test_crop_ir(self):
        assert keras_checkpoint_name("crop_ir") == "best_crop_ir_fixed.keras"

    def test_mobilefacenet_crop_ir_is_separate_from_baseline(self):
        assert keras_checkpoint_name("crop_ir", "mobilefacenet") == "best_crop_ir_mobilefacenet_fixed.keras"

    def test_full_path(self):
        p = keras_checkpoint_path("model/keras", "crop_ir")
        assert p == os.path.join("model/keras", "best_crop_ir_fixed.keras")


# ── TFLite 이름 ───────────────────────────────────────────────────

class TestTfliteNaming:
    def test_name(self):
        assert tflite_name("best_crop_ir_fixed", "npu_int8") == "best_crop_ir_fixed_npu_int8.tflite"

    def test_path(self):
        p = tflite_path("model/keras", "best_model_fixed", "float")
        assert p == os.path.join("model/keras", "best_model_fixed_float.tflite")

    def test_paths_all_variants(self):
        result = tflite_paths("model/keras", "dual")
        assert set(result.keys()) == set(TFLITE_VARIANTS)
        assert result["int8"].endswith("best_model_fixed_int8.tflite")

    def test_paths_selected_variants(self):
        result = tflite_paths("model/keras", "crop_rgb", variants=("float",))
        assert len(result) == 1
        assert "float" in result

    def test_mobilefacenet_tflite_paths_are_separate_from_baseline(self):
        result = tflite_paths("model/keras", "crop_ir", backbone="mobilefacenet")
        assert result["npu_int8"].endswith("best_crop_ir_mobilefacenet_fixed_npu_int8.tflite")


# ── 매니페스트 이름 ───────────────────────────────────────────────

class TestManifestNaming:
    def test_sidecar_manifest_path(self):
        p = sidecar_manifest_path("model/keras/best_model_fixed_int8.tflite")
        assert p == "model/keras/best_model_fixed_int8_manifest.json"

    def test_calibration_manifest_name(self):
        n = calibration_manifest_name("best_crop_ir_fixed")
        assert n == "best_crop_ir_fixed_calibration_manifest.json"

    def test_calibration_manifest_path(self):
        p = calibration_manifest_path("model/keras", "best_crop_ir_fixed")
        assert p == os.path.join("model/keras", "best_crop_ir_fixed_calibration_manifest.json")


# ── 학습곡선과 메타데이터 이름 ────────────────────────────────────

class TestCurvesAndMetadata:
    def test_curves_dual(self):
        assert learning_curves_name("dual") == "learning_curves_fixed.png"

    def test_curves_single(self):
        assert learning_curves_name("crop_rgb") == "learning_curves_crop_rgb_fixed.png"

    def test_curves_path(self):
        p = learning_curves_path("model/keras", "crop_ir")
        assert p == os.path.join("model/keras", "learning_curves_crop_ir_fixed.png")

    def test_mobilefacenet_curves_are_separate_from_baseline(self):
        assert learning_curves_name("crop_ir", "mobilefacenet") == "learning_curves_crop_ir_mobilefacenet_fixed.png"

    def test_metadata_name(self):
        assert metadata_name("20260716T120000Z_dual") == "20260716T120000Z_dual_metadata.json"

    def test_metadata_path(self):
        p = metadata_path("model/keras", "20260716T120000Z_dual")
        assert p == os.path.join("model/keras", "20260716T120000Z_dual_metadata.json")


# ── 덮어쓰기 방지 ─────────────────────────────────────────────────

class TestOverwriteProtection:
    def test_no_existing_file_passes(self, tmp_path):
        check_no_overwrite(str(tmp_path / "nonexistent.tflite"))

    def test_existing_file_raises(self, tmp_path):
        target = tmp_path / "existing.tflite"
        target.write_text("dummy")
        with pytest.raises(FileExistsError, match="--force"):
            check_no_overwrite(str(target))

    def test_existing_file_with_force(self, tmp_path):
        target = tmp_path / "existing.tflite"
        target.write_text("dummy")
        check_no_overwrite(str(target), force=True)  # 예외가 발생하지 않아야 한다.


# ── 기존 규칙과의 일관성 ──────────────────────────────────────────

class TestLegacyConsistency:
    """중앙화한 이름이 리팩터링 전 셸 래퍼와 변환기의 규칙에 맞는지 검증한다."""

    def test_shell_prefix_dual(self):
        stem = os.path.splitext(keras_checkpoint_name("dual"))[0]
        assert stem == "best_model_fixed"
        assert tflite_name(stem, "float") == "best_model_fixed_float.tflite"
        assert tflite_name(stem, "int8") == "best_model_fixed_int8.tflite"
        assert tflite_name(stem, "npu_int8") == "best_model_fixed_npu_int8.tflite"

    def test_shell_prefix_crop_rgb(self):
        stem = os.path.splitext(keras_checkpoint_name("crop_rgb"))[0]
        assert stem == "best_crop_rgb_fixed"
        assert tflite_name(stem, "float") == "best_crop_rgb_fixed_float.tflite"

    def test_shell_prefix_crop_ir(self):
        stem = os.path.splitext(keras_checkpoint_name("crop_ir"))[0]
        assert stem == "best_crop_ir_fixed"
        assert tflite_name(stem, "npu_int8") == "best_crop_ir_fixed_npu_int8.tflite"

    def test_default_output_dir(self):
        assert DEFAULT_OUTPUT_DIR == "model/keras"
