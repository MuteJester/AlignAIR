"""The controller's progress drives the gym; an evaluator that always passes the
gate climbs floors as fit() runs. Uses a stub trainer to stay GPU-free."""
from alignair.train.gym.control import GymConfig, GateSpec, RankLadder, PromotionGate, GymController


class _StubGym:
    def __init__(self):
        self.progress_calls = []
    def set_progress(self, p):
        self.progress_calls.append(p)


def test_controller_progress_climbs_with_passing_exams():
    gates = (GateSpec("v_call", "higher", (0.8, 0.8, 0.8)),)
    cfg = GymConfig(n_levels=3, gates=gates, patience=2, exam_every=1)
    ctrl = GymController(cfg, RankLadder(n_levels=3), PromotionGate(gates),
                         evaluator=lambda level, batches: {"v_call": 0.99},
                         hud=None, emit=lambda *_: None)
    gym = _StubGym()
    # simulate fit()'s exam cadence: each exam promotes (passing metrics)
    for step in range(1, 4):
        ctrl.exam(step)
        gym.set_progress(ctrl.progress())
        if ctrl.done:
            break
    assert ctrl.best_level == 2          # climbed 0 -> 1 -> 2 (top)
    assert ctrl.done is True
    assert gym.progress_calls[-1] == 1.0  # top floor progress
