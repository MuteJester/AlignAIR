from alignair.gym.control.rigor import (
    mann_kendall_trend, HardenedCeiling, RegressionGuard)


def test_trend_up_down_flat():
    assert mann_kendall_trend([0.1, 0.2, 0.35, 0.5, 0.7]) == "up"
    assert mann_kendall_trend([0.7, 0.5, 0.35, 0.2, 0.1]) == "down"
    assert mann_kendall_trend([0.4, 0.5, 0.6, 0.5, 0.4]) == "flat"   # rise-then-fall, S=0


def test_short_series_is_flat():
    assert mann_kendall_trend([0.5]) == "flat"


def test_ceiling_improving_then_ceiling():
    c = HardenedCeiling(window=4, eps=0.05)
    for v in [0.4, 0.5, 0.6, 0.7]:
        assert c.update(v) == "improving"
    last = None
    for v in [0.70, 0.701, 0.699, 0.70]:           # flat
        last = c.update(v)
    assert last == "ceiling"


def test_stall_below_floor_is_not_ceiling():
    c = HardenedCeiling(window=4, eps=0.05, floor=0.9)
    last = None
    for v in [0.5, 0.5, 0.5, 0.5, 0.5]:            # flat, but below floor 0.9
        last = c.update(v)
    assert last == "stall"                         # sampler stalled, not a capacity ceiling


def test_regression_guard_flags_drops():
    g = RegressionGuard(margin=0.03)
    assert g.check({"a": {"lo": 0.80}, "b": {"lo": 0.50}}) == []   # first sight, no baseline
    assert g.check({"a": {"lo": 0.81}, "b": {"lo": 0.52}}) == []   # both improved
    drop = g.check({"a": {"lo": 0.70}, "b": {"lo": 0.52}})         # a fell 0.81->0.70 (>0.03)
    assert drop == ["a"]
