"""Centralized artifact path and filename conventions for the Keras pipeline.

Every script that produces or consumes a model artifact (tf_train, convert,
evaluate, shell wrappers) should derive filenames from this module instead
of duplicating naming rules inline.
"""
import os
from pathlib import Path

# Default output directory (matches existing --output-dir default)
DEFAULT_OUTPUT_DIR = "model/keras"

# TFLite variant names, in canonical order
TFLITE_VARIANTS = ("float", "int8", "npu_int8")


# ---------------------------------------------------------------------------
# Keras checkpoint naming
# ---------------------------------------------------------------------------

def keras_checkpoint_name(model_type: str) -> str:
    """Return the canonical .keras checkpoint filename for a model type."""
    # dual만 접미사가 없다. 이 파이프라인이 dual 하나로 시작했을 때 굳은 이름이라
    # 기존 산출물·문서·안드로이드 자산과의 호환을 위해 그대로 유지한다.
    if model_type == "dual":
        return "best_model_fixed.keras"
    return f"best_{model_type}_fixed.keras"


def keras_checkpoint_path(output_dir: str, model_type: str) -> str:
    """Return the full path to the .keras checkpoint."""
    return os.path.join(output_dir, keras_checkpoint_name(model_type))


# ---------------------------------------------------------------------------
# TFLite naming
# ---------------------------------------------------------------------------

def tflite_name(keras_stem: str, variant: str) -> str:
    """Return the TFLite filename for a given Keras stem and variant.

    >>> tflite_name("best_crop_ir_fixed", "npu_int8")
    'best_crop_ir_fixed_npu_int8.tflite'
    """
    return f"{keras_stem}_{variant}.tflite"


def tflite_path(output_dir: str, keras_stem: str, variant: str) -> str:
    """Return the full path to a TFLite artifact."""
    return os.path.join(output_dir, tflite_name(keras_stem, variant))


def tflite_paths(output_dir: str, model_type: str, variants=None):
    """Return a dict of {variant: full_path} for the requested variants."""
    if variants is None:
        variants = TFLITE_VARIANTS
    stem = Path(keras_checkpoint_name(model_type)).stem
    return {v: tflite_path(output_dir, stem, v) for v in variants}


# ---------------------------------------------------------------------------
# Manifest / sidecar naming
# ---------------------------------------------------------------------------

def sidecar_manifest_path(tflite_file_path: str) -> str:
    """Return the sidecar manifest path for a TFLite file."""
    return tflite_file_path.replace(".tflite", "_manifest.json")


def calibration_manifest_name(keras_stem: str) -> str:
    """Return the calibration manifest filename."""
    return f"{keras_stem}_calibration_manifest.json"


def calibration_manifest_path(output_dir: str, keras_stem: str) -> str:
    """Return the full path to the calibration manifest."""
    return os.path.join(output_dir, calibration_manifest_name(keras_stem))


# ---------------------------------------------------------------------------
# Learning curves / metadata
# ---------------------------------------------------------------------------

def learning_curves_name(model_type: str) -> str:
    """Return the learning curves image filename."""
    # 체크포인트와 같은 규칙: dual만 접미사 없음.
    suffix = f"_{model_type}" if model_type != "dual" else ""
    return f"learning_curves{suffix}_fixed.png"


def learning_curves_path(output_dir: str, model_type: str) -> str:
    """Return the full path to the learning curves PNG."""
    return os.path.join(output_dir, learning_curves_name(model_type))


def metadata_name(run_id: str) -> str:
    """Return the run metadata filename."""
    return f"{run_id}_metadata.json"


def metadata_path(output_dir: str, run_id: str) -> str:
    """Return the full path to the run metadata JSON."""
    return os.path.join(output_dir, metadata_name(run_id))


# ---------------------------------------------------------------------------
# Overwrite protection
# ---------------------------------------------------------------------------

def check_no_overwrite(path: str, force: bool = False) -> None:
    """Raise FileExistsError if *path* exists and *force* is False.

    Call this before writing any artifact to prevent accidental overwrites
    of previous training/conversion results.

    tf_train.py는 fit()을 시작하기 '전에' 이 검사를 호출한다. 몇 시간 학습한 뒤
    저장 단계에서 실패해 결과를 통째로 잃는 상황을 막기 위한 순서다.
    """
    if os.path.exists(path) and not force:
        raise FileExistsError(
            f"Artifact already exists and --force was not specified: {path}"
        )
