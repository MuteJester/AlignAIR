from alignair.aa.guard.thresholds import (clopper_pearson_upper, passes_guardband, select_thresholds)


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


def test_select_thresholds_picks_first_admissible_grid_point():
    grid = [
        ({"cut": 0.9}, 3, 74, 0, 74),      # far_k=3/74 -> upper ~0.10 > 0.04 -> rejected
        ({"cut": 0.5}, 0, 74, 0, 74),      # zero-error at 74 -> passes
    ]
    t = select_thresholds("C2", "sel", grid, gate_kind="gate_5pct")
    assert t is not None and t.grid_point == {"cut": 0.5} and t.far_upper <= 0.04


def test_select_returns_none_when_no_point_passes():
    grid = [({"cut": 0.9}, 5, 74, 0, 74)]
    assert select_thresholds("C2", "sel", grid, gate_kind="gate_5pct") is None
