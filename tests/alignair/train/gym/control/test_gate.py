from alignair.train.gym.control.config import GateSpec
from alignair.train.gym.control.gate import PromotionGate


def _gate():
    return PromotionGate((
        GateSpec("v_call", "higher", (0.90, 0.80)),
        GateSpec("coords_mae", "lower", (2.0, 4.0)),
    ))


def test_all_pass_promotes():
    ok, blocking = _gate().evaluate({"v_call": 0.95, "coords_mae": 1.5}, level=0)
    assert ok is True
    assert blocking == []


def test_one_closed_blocks_and_names_it():
    ok, blocking = _gate().evaluate({"v_call": 0.85, "coords_mae": 1.5}, level=0)
    assert ok is False
    assert blocking == ["v_call"]


def test_thresholds_relax_by_level():
    g = _gate()
    # v_call 0.85 fails level 0 (bar 0.90) but passes level 1 (bar 0.80)
    assert g.evaluate({"v_call": 0.85, "coords_mae": 1.0}, level=1)[0] is True


def test_missing_metric_is_closed_not_promoted():
    ok, blocking = _gate().evaluate({"v_call": 0.95}, level=0)   # coords_mae absent
    assert ok is False
    assert "coords_mae" in blocking
