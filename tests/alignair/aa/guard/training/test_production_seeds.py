import pytest

from alignair.aa.guard.training import choose_production_seed
from alignair.errors import ConfigurationError, InputError

pytestmark = pytest.mark.component


def test_minimum_loss_wins_regardless_of_seed_or_insertion_order():
    results = {0: {"dev_loss": 0.5}, 8: {"dev_loss": 0.1}, 2: {"dev_loss": 0.3}}
    assert choose_production_seed(results) == 8
    assert choose_production_seed(dict(reversed(list(results.items())))) == 8


def test_ties_choose_lowest_seed():
    results = {9: {"dev_loss": 0.3}, 2: {"dev_loss": 0.3}, 0: {"dev_loss": 0.5}}
    assert choose_production_seed(results) == 2


@pytest.mark.parametrize("field", [
    "fresh_accuracy", "selection_loss", "loco_loss", "blind_accuracy", "far_ood_rate",
    "far-ood_rate", "sealed_accuracy", "locked_loss", "TEST_accuracy",
])
def test_fresh_metrics_are_rejected_even_for_a_losing_seed(field):
    results = {0: {"dev_loss": 0.1}, 1: {"dev_loss": 0.9, field: 0.2}}
    with pytest.raises(InputError, match="fresh-validation") as exc:
        choose_production_seed(results)
    assert isinstance(exc.value, ValueError)
    assert exc.value.context["seed"] == 1


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), -float("inf"), None, "0.1", True])
def test_invalid_loss_cannot_determine_a_production_seed(loss):
    with pytest.raises(InputError, match="finite"):
        choose_production_seed({0: {"dev_loss": 0.1}, 1: {"dev_loss": loss}})


@pytest.mark.parametrize("results", [{}, {0: {}}, {0: None}, {"0": {"dev_loss": 0.1}}])
def test_missing_or_malformed_development_results_raise_domain_errors(results):
    with pytest.raises(InputError):
        choose_production_seed(results)


def test_unknown_frozen_rule_is_not_silently_ignored():
    with pytest.raises(ConfigurationError) as exc:
        choose_production_seed({0: {"dev_loss": 0.1}}, frozen_rule="highest_accuracy")
    assert exc.value.context["rule"] == "highest_accuracy"
