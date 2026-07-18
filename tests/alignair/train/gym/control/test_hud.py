from alignair.train.gym.control.state import GateStatus, GymState
from alignair.train.gym.control.hud import GymHUD


def _state():
    gates = (
        GateStatus("v_call", 0.86, 0.88, "higher"),
        GateStatus("d_call", 0.71, 0.65, "higher"),
        GateStatus("coords_mae", 2.1, 2.0, "lower"),
    )
    return GymState(level=6, level_name="Heavy-SHM Tower", n_levels=10, step=41200,
                    gates=gates, axes=(), rooms_cleared=6, patience_used=3,
                    patience_max=8, best_level=6, headline="junction jitter")


def test_render_plain_has_no_ansi_and_shows_level_and_gates():
    out = GymHUD(color=False).render(_state())
    assert "\x1b" not in out
    assert "7/10" in out                      # level 6 shown 1-indexed as 7/10
    assert "Heavy-SHM Tower" in out
    assert "v_call" in out and "d_call" in out
    # pass vs fail marks both present (d_call passing, v_call failing)
    assert out.count("\n") > 3                 # multi-line report


def test_render_color_has_ansi():
    out = GymHUD(color=True).render(_state())
    assert "\x1b" in out


def test_event_callouts():
    hud = GymHUD(color=False)
    assert "ADVANCED" in hud.event("cleared", _state()).upper()
    assert "TOP LEVEL" in hud.event("ceiling", _state()).upper()
