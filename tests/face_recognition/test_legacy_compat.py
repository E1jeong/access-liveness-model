from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.tensorflow
def test_legacy_export_script_direct_execution_help():
    """Verify scripts/recognition/export_npu_mobilenet_emore.py runs without ModuleNotFoundError."""
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "recognition" / "export_npu_mobilenet_emore.py"

    res = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    assert res.returncode == 0, f"STDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert "Export NPU-compliant INT8 TFLite" in res.stdout


def test_legacy_calibration_module_import():
    """Verify scripts/recognition/calibration.py imports and re-exports cleanly."""
    repo_root = Path(__file__).resolve().parents[2]

    res = subprocess.run(
        [
            sys.executable,
            "-c",
            "import scripts.recognition.calibration as c; print(c.CALIBRATION_SAMPLES)",
        ],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    assert res.returncode == 0, f"STDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert "500" in res.stdout
