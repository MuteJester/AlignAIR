from alignair.train.gym.control.config import GateSpec, GymConfig, default_gates, _linspace


def test_linspace_endpoints_and_length():
    xs = _linspace(0.97, 0.80, 10)
    assert len(xs) == 10
    assert xs[0] == 0.97
    assert abs(xs[-1] - 0.80) < 1e-9


def test_default_gates_cover_heads_and_relax_with_level():
    gates = default_gates(10)
    by = {g.metric: g for g in gates}
    assert set(by) == {"v_call", "d_call", "j_call", "coords_mae", "region_acc"}
    # higher-better gates get EASIER (lower bar) at harder levels
    assert by["v_call"].thresholds[0] > by["v_call"].thresholds[-1]
    # lower-better MAE gate ALLOWS more error at harder levels
    assert by["coords_mae"].direction == "lower"
    assert by["coords_mae"].thresholds[0] < by["coords_mae"].thresholds[-1]
    # every gate has one threshold per level
    assert all(len(g.thresholds) == 10 for g in gates)


def test_gymconfig_defaults():
    cfg = GymConfig()
    assert cfg.n_levels == 10
    assert len(cfg.gates) == 5
    assert cfg.patience == 8
