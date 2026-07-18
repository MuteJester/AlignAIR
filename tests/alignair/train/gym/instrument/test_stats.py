from alignair.train.gym.instrument.stats import bootstrap_ci


def test_ci_brackets_mean_and_is_deterministic():
    vals = [1.0] * 50 + [0.0] * 50      # mean 0.5
    m, lo, hi = bootstrap_ci(vals, n_boot=500, seed=7)
    assert abs(m - 0.5) < 1e-9
    assert lo < m < hi
    assert (lo, hi) == bootstrap_ci(vals, n_boot=500, seed=7)[1:]   # deterministic


def test_constant_values_give_zero_width():
    m, lo, hi = bootstrap_ci([0.8] * 20, n_boot=200, seed=1)
    assert abs(m - 0.8) < 1e-9       # not bit-exact 0.8 (float summation)
    assert hi - lo < 1e-12           # constant input => zero-WIDTH interval


def test_empty_is_zero():
    assert bootstrap_ci([], n_boot=10, seed=1) == (0.0, 0.0, 0.0)
