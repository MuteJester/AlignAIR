import pytest
from alignair.aa.guard.thresholds import (
    ceiling_for, clopper_pearson_upper, far_limit, passes_guardband,
)

pytestmark = pytest.mark.component


def test_clopper_pearson_zero_error_floors():
    assert clopper_pearson_upper(0, 74) <= 0.04
    assert clopper_pearson_upper(0, 59) <= 0.05
    assert clopper_pearson_upper(0, 36) <= 0.08
    assert clopper_pearson_upper(0, 35) > 0.08          # 35 is below the 10%-gate zero-error floor
    assert clopper_pearson_upper(0, 0) == 1.0 and clopper_pearson_upper(5, 5) == 1.0


def test_passes_guardband_rejects_bound_over_ceiling():
    assert passes_guardband(0.039, 0.07, "gate_5pct")
    assert not passes_guardband(0.05, 0.07, "gate_5pct")     # FAR upper over 4%
    assert not passes_guardband(0.039, 0.09, "gate_5pct")    # FRR upper over 8%
    assert passes_guardband(0.079, 0.079, "gate_10pct")


def test_limits_and_ceilings():
    assert far_limit("gate_5pct") == 0.04 and far_limit("gate_10pct") == 0.08
    assert ceiling_for("far", "gate_5pct") == 0.04 and ceiling_for("frr", "frr") == 0.08
