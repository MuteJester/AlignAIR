import random
from alignair.train.gym.instrument.task_space import Axis, TaskSpace


def test_deployment_box_covers_hard_tail():
    ts = TaskSpace.deployment()
    by = {a.name: a for a in ts.axes}
    assert by["mutation_rate"].hi >= 0.25      # hard tail, NOT capped at 0.15
    assert by["end_loss_5"].hi >= 100
    assert "crop_len" in by and "orient_prob" in by


def test_unfixed_axes_take_baseline_and_are_deterministic():
    ts = TaskSpace.deployment()
    a = ts.sample()
    b = ts.sample()
    assert a == b                               # deterministic point
    # every unfixed axis sits at its easy baseline (lo) — cells isolate named difficulty
    for ax in ts.axes:
        assert a[ax.name] == ax.lo


def test_to_genairr_params_has_required_keys():
    ts = TaskSpace.deployment()
    theta = ts.sample(random.Random(1))
    p = ts.to_genairr_params(theta)
    for k in ("mutation_rate", "end_loss_5", "end_loss_3", "indel_count",
              "seq_error_rate", "ambiguous_count", "crop_prob", "crop_len_min",
              "crop_len_max", "orient_prob"):
        assert k in p
    assert isinstance(p["end_loss_5"], tuple) and len(p["end_loss_5"]) == 2


def test_frac_fixes_axis_to_fraction_of_range():
    ts = TaskSpace.deployment()
    theta = ts.sample(random.Random(2), frac={"mutation_rate": 1.0})
    by = {a.name: a for a in ts.axes}
    assert abs(theta["mutation_rate"] - by["mutation_rate"].hi) < 1e-9
