"""Keras 파이프라인 산출물의 경로와 파일명 규칙을 한곳에서 관리한다.

모델 산출물을 만들거나 읽는 모든 코드(tf_train, 변환기, 평가기, 셸 래퍼)는
파일명 규칙을 각자 중복해서 적지 않고 이 모듈에서 가져와야 한다.
"""
import os
from pathlib import Path

# 기본 출력 디렉터리. 각 명령의 --output-dir 기본값과 같아야 한다.
DEFAULT_OUTPUT_DIR = "model/keras"

# TFLite 변형 이름. 이 튜플의 순서가 표준 출력 순서다.
TFLITE_VARIANTS = ("float", "int8", "npu_int8")


# ---------------------------------------------------------------------------
# Keras 체크포인트 이름
# ---------------------------------------------------------------------------

def keras_checkpoint_name(model_type: str) -> str:
    """모델 종류에 해당하는 표준 .keras 체크포인트 파일명을 반환한다."""
    # dual만 접미사가 없다. 이 파이프라인이 dual 하나로 시작했을 때 굳은 이름이라
    # 기존 산출물·문서·안드로이드 자산과의 호환을 위해 그대로 유지한다.
    if model_type == "dual":
        return "best_model_fixed.keras"
    return f"best_{model_type}_fixed.keras"


def keras_checkpoint_path(output_dir: str, model_type: str) -> str:
    """출력 디렉터리와 .keras 체크포인트 파일명을 결합해 반환한다."""
    return os.path.join(output_dir, keras_checkpoint_name(model_type))


# ---------------------------------------------------------------------------
# TFLite 파일 이름
# ---------------------------------------------------------------------------

def tflite_name(keras_stem: str, variant: str) -> str:
    """Keras 파일의 확장자 없는 이름과 변형으로 TFLite 파일명을 만든다.

    >>> tflite_name("best_crop_ir_fixed", "npu_int8")
    'best_crop_ir_fixed_npu_int8.tflite'
    """
    return f"{keras_stem}_{variant}.tflite"


def tflite_path(output_dir: str, keras_stem: str, variant: str) -> str:
    """출력 디렉터리와 TFLite 파일명을 결합해 반환한다."""
    return os.path.join(output_dir, tflite_name(keras_stem, variant))


def tflite_paths(output_dir: str, model_type: str, variants=None):
    """요청한 변형별 산출물 경로를 ``{변형: 경로}`` 사전으로 반환한다."""
    if variants is None:
        variants = TFLITE_VARIANTS
    stem = Path(keras_checkpoint_name(model_type)).stem
    return {v: tflite_path(output_dir, stem, v) for v in variants}


# ---------------------------------------------------------------------------
# 매니페스트와 사이드카 파일 이름
# ---------------------------------------------------------------------------

def sidecar_manifest_path(tflite_file_path: str) -> str:
    """TFLite 파일과 짝을 이루는 사이드카 매니페스트 경로를 반환한다."""
    return tflite_file_path.replace(".tflite", "_manifest.json")


def calibration_manifest_name(keras_stem: str) -> str:
    """캘리브레이션 매니페스트 파일명을 반환한다."""
    return f"{keras_stem}_calibration_manifest.json"


def calibration_manifest_path(output_dir: str, keras_stem: str) -> str:
    """출력 디렉터리와 캘리브레이션 매니페스트 파일명을 결합해 반환한다."""
    return os.path.join(output_dir, calibration_manifest_name(keras_stem))


# ---------------------------------------------------------------------------
# 학습곡선과 실행 메타데이터
# ---------------------------------------------------------------------------

def learning_curves_name(model_type: str) -> str:
    """학습곡선 이미지 파일명을 반환한다."""
    # 체크포인트와 같은 규칙: dual만 접미사 없음.
    suffix = f"_{model_type}" if model_type != "dual" else ""
    return f"learning_curves{suffix}_fixed.png"


def learning_curves_path(output_dir: str, model_type: str) -> str:
    """출력 디렉터리와 학습곡선 PNG 파일명을 결합해 반환한다."""
    return os.path.join(output_dir, learning_curves_name(model_type))


def metadata_name(run_id: str) -> str:
    """학습 실행 메타데이터 파일명을 반환한다."""
    return f"{run_id}_metadata.json"


def metadata_path(output_dir: str, run_id: str) -> str:
    """출력 디렉터리와 실행 메타데이터 JSON 파일명을 결합해 반환한다."""
    return os.path.join(output_dir, metadata_name(run_id))


# ---------------------------------------------------------------------------
# 덮어쓰기 방지
# ---------------------------------------------------------------------------

def check_no_overwrite(path: str, force: bool = False) -> None:
    """경로가 이미 있고 ``force``가 거짓이면 ``FileExistsError``를 발생시킨다.

    이전 학습·변환 결과를 실수로 덮어쓰지 않도록 산출물을 쓰기 전에 호출한다.

    tf_train.py는 fit()을 시작하기 '전에' 이 검사를 호출한다. 몇 시간 학습한 뒤
    저장 단계에서 실패해 결과를 통째로 잃는 상황을 막기 위한 순서다.
    """
    if os.path.exists(path) and not force:
        raise FileExistsError(
            f"Artifact already exists and --force was not specified: {path}"
        )
