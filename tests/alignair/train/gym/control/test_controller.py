from alignair.train.gym.control.config import GymConfig, GateSpec
from alignair.train.gym.control.ladder import RankLadder
from alignair.train.gym.control.gate import PromotionGate
from alignair.train.gym.control.controller import GymController


def _controller(evaluator, n_levels=3):
    gates = (GateSpec("v_call", "higher", tuple(0.8 for _ in range(n_levels))),)
    cfg = GymConfig(n_levels=n_levels, gates=gates, patience=2)
    return GymController(cfg, RankLadder(n_levels=n_levels),
                         PromotionGate(gates), evaluator, hud=None, emit=lambda *_: None)


def test_promotes_when_gate_passes():
    ctrl = _controller(lambda level, batches: {"v_call": 0.95})
    ctrl.exam(step=10)
    assert ctrl.level == 1
    assert ctrl.done is False


def test_clearing_top_floor_completes():
    ctrl = _controller(lambda level, batches: {"v_call": 0.95}, n_levels=2)
    ctrl.exam(step=1)     # 0 -> 1
    ctrl.exam(step=2)     # 1 (top) cleared -> complete
    assert ctrl.level == 1
    assert ctrl.done is True


def test_plateau_sets_ceiling():
    ctrl = _controller(lambda level, batches: {"v_call": 0.5})  # never passes (bar 0.8)
    ctrl.exam(step=1)     # composite ~0.625, best
    ctrl.exam(step=2)     # stall 1
    ctrl.exam(step=3)     # stall 2 == patience -> ceiling
    assert ctrl.level == 0
    assert ctrl.done is True


def test_coords_mae_is_derived_from_e2e_devs():
    def ev(level, batches):
        return {"v_call": 0.99, "v_e2e_gl_start_dev": 1.0, "v_e2e_gl_end_dev": 3.0}
    gates = (GateSpec("coords_mae", "lower", (2.5,)),)
    cfg = GymConfig(n_levels=1, gates=gates, patience=2)
    ctrl = GymController(cfg, RankLadder(n_levels=1), PromotionGate(gates), ev,
                         hud=None, emit=lambda *_: None)
    state = ctrl.exam(step=1)
    # mean(1.0, 3.0) = 2.0 <= 2.5 => coords gate open
    cm = next(g for g in state.gates if g.name == "coords_mae")
    assert abs(cm.value - 2.0) < 1e-9 and cm.is_open
