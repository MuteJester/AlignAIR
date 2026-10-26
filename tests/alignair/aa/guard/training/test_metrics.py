import pytest

from alignair.aa.guard.training.metrics import binary_auroc
from alignair.errors import DataContractError

pytestmark = pytest.mark.component


@pytest.mark.parametrize("scores, expected", [([0, 0, 1, 1], 1), ([1, 1, 0, 0], 0),
                                             ([1, 1, 1, 1], 0.5), ([0, 1, 1, 2], 0.875)])
def test_auroc_orders_positives_and_gives_ties_half_credit(scores, expected):
    assert binary_auroc([0, 0, 1, 1], scores) == expected


def test_single_class_auroc_is_unavailable():
    assert binary_auroc([1, 1], [0.1, 0.2]) is None


@pytest.mark.parametrize("labels, scores", [([0, 1], [1]), ([0, 2], [0, 1]), ([0, 1], [0, float("nan")])])
def test_invalid_metric_inputs_are_reported(labels, scores):
    with pytest.raises(DataContractError):
        binary_auroc(labels, scores)
