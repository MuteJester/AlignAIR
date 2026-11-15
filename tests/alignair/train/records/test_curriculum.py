import math

import pytest
from alignair.errors import ConfigurationError
from alignair.train.records.curriculum import Curriculum
from alignair.train.records.curriculum import INDEL_COUNT_WEIGHTS, INDEL_FRACTION

pytestmark = pytest.mark.component


def test_curriculum_ramps_and_describes():
    c = Curriculum()
    easy = c.params(0.0)
    hard = c.params(1.0)
    # harder at p=1: higher mutation rate cap, more trim, indels, seq errors
    assert hard["mutation_rate"] >= easy["mutation_rate"]
    assert hard["end_loss_5"][1] >= easy["end_loss_5"][1]
    assert hard["indel_count"][1] >= easy["indel_count"][1]
    assert hard["seq_error_rate"] >= easy["seq_error_rate"]
    d = c.describe(0.5)
    assert isinstance(d, str) and "stage" in d.lower()
    assert "indel≤2" in d
    assert f"indel {100 * sum(INDEL_FRACTION) / 2:.0f}%" in Curriculum(indel_form="weighted").describe(0.5)


def test_curriculum_stage_index():
    c = Curriculum(stages=5)
    assert c.stage(0.0) == 0
    assert c.stage(1.0) == 4


@pytest.mark.parametrize("p", [-1.0, 0.0, 0.1, 0.3, 0.5, 0.6, 0.9, 1.0, 2.0])
def test_the_default_indel_form_is_the_historical_uniform_count_range(p):
    # Benchmark, evaluation and matcher datasets are generated from this form; it must not drift.
    assert Curriculum().indel_form == "uniform"
    assert Curriculum().params(p)["indel_count"] == (0, int(round(5 * min(1.0, max(0.0, p)))))


@pytest.mark.parametrize("p", [-1.0, 0.0, 0.3, 0.6, 0.9, 1.0, 2.0])
def test_the_weighted_indel_form_draws_from_weights_whose_indel_share_ramps_with_progress(p):
    low, high = INDEL_FRACTION
    share = low + (high - low) * min(1.0, max(0.0, p))
    pairs = Curriculum(indel_form="weighted").params(p)["indel_count"]
    weights = dict(pairs)
    assert [count for count, _ in pairs] == [0, 1, 2, 3]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights[0] == pytest.approx(1.0 - share)
    # among the weights of reads with indels: one dominates, then two, then three -- at every progress
    assert weights[1] > weights[2] > weights[3] > 0
    for count, weight in INDEL_COUNT_WEIGHTS:
        assert weights[count] / share == pytest.approx(weight)


def test_indel_fraction_is_a_curriculum_parameter_and_zero_weights_are_left_out():
    assert Curriculum().indel_fraction == INDEL_FRACTION
    weighted = lambda fraction: Curriculum(indel_form="weighted", indel_fraction=fraction)
    assert dict(weighted((0.1, 0.3)).params(0.5)["indel_count"])[0] == pytest.approx(0.8)
    assert weighted((0, 0)).params(1.0)["indel_count"] == ((0, 1.0),)
    assert [count for count, _ in weighted((1, 1)).params(0.0)["indel_count"]] == [1, 2, 3]


@pytest.mark.parametrize("value", [None, 0.2, "ab", (0.1,), (0.1, 0.2, 0.3), (0.3, 0.1), (-0.1, 0.2), (0.1, 1.5),
                                   (True, 0.2), (math.nan, 0.2), (0.1, math.inf)])
def test_indel_fraction_must_be_a_nondecreasing_range_of_probabilities(value):
    with pytest.raises(ConfigurationError) as caught:
        Curriculum(indel_fraction=value)
    assert caught.value.context["field"] == "indel_fraction"


@pytest.mark.parametrize("value", [None, "", "Weighted", "realistic", 1])
def test_indel_form_is_uniform_or_weighted(value):
    with pytest.raises(ConfigurationError) as caught:
        Curriculum(indel_form=value)
    assert caught.value.context["field"] == "indel_form"


def test_sequencing_errors_stay_at_the_level_the_chip_trained_on():
    from alignair.train.records.curriculum import SEQUENCING_ERROR_SCALE
    assert SEQUENCING_ERROR_SCALE == 0.757
    assert Curriculum().params(1.0)["seq_error_rate"] == pytest.approx(0.02 * 0.757)
    assert Curriculum().params(0.5)["seq_error_rate"] == pytest.approx(0.01 * 0.757)
    assert Curriculum().params(0.0)["seq_error_rate"] == 0.0
