import json

import numpy as np
import pytest

from classes import CLASS_NAMES
from evaluate_tflite import write_metrics_csv, write_regression_report


def _result(name, logits, acer, latency):
    logits = np.asarray(logits, dtype=np.float32)
    return {
        "name": name,
        "model": f"model/{name}",
        "accuracy": 1.0,
        "apcer": acer,
        "bpcer": acer,
        "acer": acer,
        "recalls": [1.0] * len(CLASS_NAMES),
        "mean_latency_ms": latency,
        "file_size_bytes": 10,
        "_labels": np.array([0, 1]),
        "_preds": np.argmax(logits, axis=1),
        "_logits": logits,
    }


def test_regression_report_records_logits_metrics_size_and_latency(tmp_path):
    baseline = _result("keras", [[3, 0], [0, 3]], acer=0.1, latency=1.0)
    artifact = _result("int8", [[2, 0], [1, 2]], acer=0.2, latency=1.5)
    path = tmp_path / "regression.json"

    report = write_regression_report([baseline, artifact], path, split="validation")

    saved = json.loads(path.read_text())
    assert saved == report
    assert report["baseline"] == "keras"
    assert report["split"] == "validation"
    assert report["acer_policy"] == "report_only"
    assert report["comparisons"] == [{
        "artifact": "int8",
        "logits_max_abs_error": 1.0,
        "logits_mean_abs_error": 0.75,
        "argmax_agreement": 1.0,
        "acer_delta": 0.1,
        "mean_latency_ms_delta": 0.5,
    }]


def test_regression_report_rejects_misaligned_labels(tmp_path):
    baseline = _result("keras", [[3, 0], [0, 3]], acer=0.1, latency=1.0)
    artifact = _result("int8", [[2, 0], [1, 2]], acer=0.2, latency=1.5)
    artifact["_labels"] = np.array([1, 0])

    with pytest.raises(ValueError, match="label 순서"):
        write_regression_report([baseline, artifact], tmp_path / "regression.json")


def test_metrics_csv_records_an_explicit_test_split(tmp_path):
    path = tmp_path / "test.csv"
    write_metrics_csv([_result("keras", [[3, 0], [0, 3]], acer=0.1, latency=1.0)], path, "test")

    assert path.read_text().splitlines()[0].startswith("split,name,model")
    assert path.read_text().splitlines()[1].startswith("test,keras,")
