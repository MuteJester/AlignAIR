import math

import pytest
from alignair.errors import ConfigurationError
from alignair.train.gym.curriculum import Curriculum
from alignair.train.gym.curriculum.base import INDEL_COUNT_WEIGHTS, INDEL_FRACTION

pytestmark = pytest.mark.component


def test_curriculum_ramps_and_describes():
    c = Curriculum()
    easy = c.params(0.0)
    hard = c.params(1.0)
    # harder at p=1: higher mutation rate cap, more trim, indels, seq errors
    assert hard["mutation_rate"] >= easy["mutation_rate"]
    assert hard["end_loss_5"][1] >= easy["end_loss_5"][1]
    assert dict(hard["indel_count"])[0] < dict(easy["indel_count"])[0]
    assert hard["seq_error_rate"] >= easy["seq_error_rate"]
    d = c.describe(0.5)
    assert isinstance(d, str) and "stage" in d.lower()
    assert f"indel {100 * sum(INDEL_FRACTION) / 2:.0f}%" in d


def test_curriculum_stage_index():
    c = Curriculum(stages=5)
    assert c.stage(0.0) == 0
    assert c.stage(1.0) == 4


@pytest.mark.parametrize("p", [-1.0, 0.0, 0.3, 0.6, 0.9, 1.0, 2.0])
def test_indel_count_is_weighted_and_its_indel_share_ramps_with_progress(p):
    low, high = INDEL_FRACTION
    share = low + (high - low) * min(1.0, max(0.0, p))
    pairs = Curriculum().params(p)["indel_count"]
    weights = dict(pairs)
    assert [count for count, _ in pairs] == [0, 1, 2, 3]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights[0] == pytest.approx(1.0 - share)
    # among reads with indels, one dominates, then two, rarely three -- at every progress
    assert weights[1] > weights[2] > weights[3] > 0
    for count, weight in INDEL_COUNT_WEIGHTS:
        assert weights[count] / share == pytest.approx(weight)


def test_indel_fraction_is_a_curriculum_parameter_and_zero_weights_are_left_out():
    assert Curriculum().indel_fraction == INDEL_FRACTION
    assert dict(Curriculum(indel_fraction=(0.1, 0.3)).params(0.5)["indel_count"])[0] == pytest.approx(0.8)
    assert Curriculum(indel_fraction=(0, 0)).params(1.0)["indel_count"] == ((0, 1.0),)
    assert [count for count, _ in Curriculum(indel_fraction=(1, 1)).params(0.0)["indel_count"]] == [1, 2, 3]


@pytest.mark.parametrize("value", [None, 0.2, "ab", (0.1,), (0.1, 0.2, 0.3), (0.3, 0.1), (-0.1, 0.2), (0.1, 1.5),
                                   (True, 0.2), (math.nan, 0.2), (0.1, math.inf)])
def test_indel_fraction_must_be_a_nondecreasing_range_of_probabilities(value):
    with pytest.raises(ConfigurationError) as caught:
        Curriculum(indel_fraction=value)
    assert caught.value.context["field"] == "indel_fraction"
