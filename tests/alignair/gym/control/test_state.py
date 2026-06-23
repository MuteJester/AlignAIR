from alignair.gym.control.state import GateStatus, AxisStat, GymState, composite_score


def test_gatestatus_open_and_fraction_higher():
    g = GateStatus("v_call", value=0.86, threshold=0.88, direction="higher")
    assert g.is_open is False
    assert abs(g.fraction - 0.86 / 0.88) < 1e-9
    assert GateStatus("v_call", 0.90, 0.88, "higher").is_open is True


def test_gatestatus_open_and_fraction_lower():
    g = GateStatus("coords_mae", value=2.1, threshold=2.0, direction="lower")
    assert g.is_open is False
    assert abs(g.fraction - 2.0 / 2.1) < 1e-9
    assert GateStatus("coords_mae", 1.5, 2.0, "lower").is_open is True
    # fraction never exceeds 1 even when comfortably open
    assert GateStatus("coords_mae", 0.5, 2.0, "lower").fraction == 1.0


def test_composite_and_blocking():
    gates = (
        GateStatus("v_call", 0.90, 0.88, "higher"),     # open, frac 1.0
        GateStatus("j_call", 0.60, 0.80, "higher"),     # closed, frac 0.75
    )
    assert abs(composite_score(gates) - 0.875) < 1e-9
    st = GymState(level=3, level_name="Room 4", n_levels=10, step=100,
                  gates=gates, axes=(), rooms_cleared=3, patience_used=1,
                  patience_max=8, best_level=3, headline="")
    assert st.all_open is False
    assert st.blocking == ("j_call",)
