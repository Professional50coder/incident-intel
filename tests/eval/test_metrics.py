import pytest

from incident_intel.eval.metrics import compute_auc_roc


def test_perfect_separation_gives_auc_one() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]

    assert compute_auc_roc(labels, scores) == pytest.approx(1.0)


def test_inverted_scores_give_auc_zero() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.9, 0.8, 0.2, 0.1]

    assert compute_auc_roc(labels, scores) == pytest.approx(0.0)


def test_single_class_labels_raise_value_error() -> None:
    with pytest.raises(ValueError):
        compute_auc_roc([1, 1, 1], [0.1, 0.5, 0.9])
