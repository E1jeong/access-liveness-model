import numpy as np
import pytest

from utils import calculate_validation_metrics


def _metrics(labels, preds):
    return calculate_validation_metrics(labels, preds)[2:]


def test_all_spoof_predicted_as_live_is_full_apcer():
    apcer, bpcer, acer = _metrics([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])

    assert (apcer, bpcer, acer) == (1.0, 0.0, 0.5)


def test_all_live_predicted_as_spoof_is_full_bpcer():
    apcer, bpcer, acer = _metrics([0, 0, 0], [1, 2, 5])

    assert (apcer, bpcer, acer) == (0.0, 1.0, 0.5)


def test_spoof_subtype_misclassification_is_not_an_apcer_error():
    apcer, bpcer, acer = _metrics([1, 2, 3, 4, 5], [2, 3, 4, 5, 1])

    assert (apcer, bpcer, acer) == (0.0, 0.0, 0.0)


def test_perfect_predictions_have_zero_pad_errors():
    apcer, bpcer, acer = _metrics([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])

    assert (apcer, bpcer, acer) == (0.0, 0.0, 0.0)


def test_empty_input_has_zero_pad_errors_and_zero_recalls():
    confusion_matrix, recalls, apcer, bpcer, acer = calculate_validation_metrics([], [])

    assert np.array_equal(confusion_matrix, np.zeros((6, 6), dtype=np.int64))
    assert recalls == [0.0] * 6
    assert (apcer, bpcer, acer) == (0.0, 0.0, 0.0)


def test_metric_inputs_must_have_equal_lengths():
    with pytest.raises(ValueError, match="길이"):
        calculate_validation_metrics([0, 1], [0])


@pytest.mark.parametrize(
    ("labels", "preds", "message"),
    [([-1], [0], "labels"), ([0], [6], "preds")],
)
def test_metric_inputs_must_stay_within_class_range(labels, preds, message):
    with pytest.raises(ValueError, match=message):
        calculate_validation_metrics(labels, preds)
