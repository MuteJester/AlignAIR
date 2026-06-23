import random
from alignair.gym.instrument.task_space import Axis, TaskSpace


def test_deployment_box_covers_hard_tail():
    ts = TaskSpace.deployment()
    by = {a.name: a for a in ts.axes}
    assert by["mutation_rate"].hi >= 0.25      # hard tail, NOT capped at 0.15
    assert by["end_loss_5"].hi >= 100
    assert "crop_len" in by and "orient_prob" in by


def test_sample_in_range_and_seeded():
    ts = TaskSpace.deployment()
    a = ts.sample(random.Random(0))
    b = ts.sample(random.Random(0))
    assert a == b                               # deterministic per seed
    for ax in ts.axes:
        assert ax.lo <= a[ax.name] <= ax.hi


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
