import numpy as np
import pytest

from common.classes import ATTACK_CLASS_INDICES, BONA_FIDE_CLASS_INDICES, CLASS_NAMES
from common.utils import calculate_validation_metrics


def _metrics(labels, preds):
    return calculate_validation_metrics(labels, preds)[2:]


@pytest.mark.parametrize("bona_fide_prediction", BONA_FIDE_CLASS_INDICES)
def test_all_attacks_predicted_as_bona_fide_is_full_apcer(bona_fide_prediction):
    attack_labels = list(ATTACK_CLASS_INDICES)
    apcer, bpcer, acer = _metrics(
        attack_labels, [bona_fide_prediction] * len(attack_labels)
    )

    assert (apcer, bpcer, acer) == (1.0, 0.0, 0.5)


def test_all_bona_fide_predicted_as_attack_is_full_bpcer():
    apcer, bpcer, acer = _metrics(BONA_FIDE_CLASS_INDICES, [1, 2, 5])

    assert (apcer, bpcer, acer) == (0.0, 1.0, 0.5)


def test_attack_subtype_misclassification_is_not_an_apcer_error():
    attack_labels = list(ATTACK_CLASS_INDICES)
    apcer, bpcer, acer = _metrics(
        attack_labels, attack_labels[1:] + attack_labels[:1]
    )

    assert (apcer, bpcer, acer) == (0.0, 0.0, 0.0)


def test_bona_fide_subtype_misclassification_is_not_a_bpcer_error():
    bona_fide_labels = list(BONA_FIDE_CLASS_INDICES)
    apcer, bpcer, acer = _metrics(
        bona_fide_labels, bona_fide_labels[1:] + bona_fide_labels[:1]
    )

    assert (apcer, bpcer, acer) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("dental_prediction", BONA_FIDE_CLASS_INDICES[1:])
def test_masked_print_predicted_as_dental_is_an_apcer_error(dental_prediction):
    apcer, bpcer, acer = _metrics([1], [dental_prediction])

    assert (apcer, bpcer, acer) == (1.0, 0.0, 0.5)


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
