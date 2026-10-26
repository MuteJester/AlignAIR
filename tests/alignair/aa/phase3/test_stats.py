import pytest
from alignair.aa.phase3.stats import clopper_pearson, coverage_decision

pytestmark = pytest.mark.component


def test_bounds_bracket_estimate():
    lo, hi = clopper_pearson(9, 10, 0.05)
    assert 0 <= lo <= 0.9 <= hi <= 1


def test_perfect_small_n_cannot_support():
    assert coverage_decision(15, 15, 0.05, allow_support=True) == "inconclusive"


def test_clear_failure():
    assert coverage_decision(10, 20, 0.05, allow_support=True) == "failed"


def test_family_endpoint_never_supports():
    assert coverage_decision(60, 60, 0.05, allow_support=False) != "supported"
