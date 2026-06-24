import numpy as np
from alignair.gym.gym import _pick_params


def test_pick_params_respects_weights():
    comps = [(0.0, {"a": 1}), (1.0, {"a": 2})]
    rng = np.random.default_rng(0)
    picks = [_pick_params(comps, rng)["a"] for _ in range(20)]
    assert set(picks) == {2}                       # zero-weight component never chosen


def test_pick_params_single_component():
    assert _pick_params([(1.0, {"x": 9})], np.random.default_rng(0)) == {"x": 9}
