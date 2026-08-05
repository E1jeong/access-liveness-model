import numpy as np
import pytest

from classes import CLASS_NAMES
from utils import calculate_validation_metrics


def _metrics(labels, preds):
    return calculate_validation_metrics(labels, preds)[2:]


def test_all_spoof_predicted_as_live_is_full_apcer():
    spoof_labels = list(range(1, len(CLASS_NAMES)))
    apcer, bpcer, acer = _metrics(spoof_labels, [0] * len(spoof_labels))

    assert (apcer, bpcer, acer) == (1.0, 0.0, 0.5)


def test_all_live_predicted_as_spoof_is_full_bpcer():
    apcer, bpcer, acer = _metrics([0, 0, 0], [1, 2, 5])

    assert (apcer, bpcer, acer) == (0.0, 1.0, 0.5)


def test_spoof_subtype_misclassification_is_not_an_apcer_error():
    spoof_labels = list(range(1, len(CLASS_NAMES)))
    apcer, bpcer, acer = _metrics(spoof_labels, spoof_labels[1:] + spoof_labels[:1])

    assert (apcer, bpcer, acer) == (0.0, 0.0, 0.0)


def test_perfect_predictions_have_zero_pad_errors():
    labels = list(range(len(CLASS_NAMES)))
    apcer, bpcer, acer = _metrics(labels, labels)

    assert (apcer, bpcer, acer) == (0.0, 0.0, 0.0)


def test_empty_input_has_zero_pad_errors_and_zero_recalls():
    confusion_matrix, recalls, apcer, bpcer, acer = calculate_validation_metrics([], [])

    assert np.array_equal(confusion_matrix, np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64))
    assert recalls == [0.0] * len(CLASS_NAMES)
    assert (apcer, bpcer, acer) == (0.0, 0.0, 0.0)


def test_metric_inputs_must_have_equal_lengths():
    with pytest.raises(ValueError, match="길이"):
        calculate_validation_metrics([0, 1], [0])


@pytest.mark.parametrize(
    ("labels", "preds", "message"),
    [([-1], [0], "labels"), ([0], [len(CLASS_NAMES)], "preds")],
)
def test_metric_inputs_must_stay_within_class_range(labels, preds, message):
    with pytest.raises(ValueError, match=message):
        calculate_validation_metrics(labels, preds)
