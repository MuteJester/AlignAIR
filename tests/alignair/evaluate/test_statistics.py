import pytest
import random
import numpy as np
from alignair.errors import ConfigurationError, InputError
from alignair.evaluate import (
    bootstrap_mean_ci, paired_diff_ci,
    wilson_interval, bootstrap_competence_ci,
)

pytestmark = pytest.mark.component


def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = wilson_interval(80, 100)
    assert lo < 0.8 < hi and 0.0 <= lo and hi <= 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)
    tight = wilson_interval(8000, 10000)
    wide = wilson_interval(8, 10)
    assert (tight[1] - tight[0]) < (wide[1] - wide[0])       # more data -> tighter


def test_bootstrap_mean_ci_is_deterministic_and_brackets_the_mean():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    a = bootstrap_mean_ci(vals, n_boot=200, seed=0)
    assert a == bootstrap_mean_ci(vals, n_boot=200, seed=0)   # seeded -> reproducible
    assert a[0] <= 3.0 <= a[1]


def test_paired_diff_ci_detects_a_real_advantage_and_rejects_a_null_one():
    # model beats baseline on 30% of reads, never loses -> lower bound must clear zero
    win = paired_diff_ci([1.0] * 300 + [0.0] * 700, n_boot=500, seed=0)
    assert win["mean"] > 0 and win["lower_gt_zero"]
    # symmetric wins/losses -> no detectable advantage
    null = paired_diff_ci(([1.0] * 500 + [-1.0] * 500), n_boot=500, seed=0)
    assert not null["lower_gt_zero"]


def test_paired_comparison_is_not_the_same_as_independent_intervals():
    """Paired differences see a consistent small advantage that overlapping marginal CIs miss."""
    n = 2000
    # model correct on 1020, baseline on 1000, and the model is right whenever the baseline is.
    diffs = [1.0] * 20 + [0.0] * (n - 20)
    paired = paired_diff_ci(diffs, n_boot=500, seed=0)
    assert paired["lower_gt_zero"]                            # paired test resolves it
    a, b = wilson_interval(1020, n), wilson_interval(1000, n)
    assert a[0] < b[1]                                        # independent intervals overlap


def test_ci_brackets_mean_and_is_deterministic():
    vals = [1.0] * 50 + [0.0] * 50      # mean 0.5
    m, lo, hi = bootstrap_competence_ci(vals, n_boot=500, seed=7)
    assert abs(m - 0.5) < 1e-9
    assert lo < m < hi
    assert (lo, hi) == bootstrap_competence_ci(vals, n_boot=500, seed=7)[1:]   # deterministic


def test_constant_values_give_zero_width():
    m, lo, hi = bootstrap_competence_ci([0.8] * 20, n_boot=200, seed=1)
    assert abs(m - 0.8) < 1e-9       # not bit-exact 0.8 (float summation)
    assert hi - lo < 1e-12           # constant input => zero-WIDTH interval


def test_empty_is_zero():
    assert bootstrap_competence_ci([], n_boot=10, seed=1) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("seed, expected", [(0, (1.16, 8.84)), (7, (2.88, 8.6))])
@pytest.mark.parametrize("budget", [1, 5, 12, 1_000_000])
def test_chunk_size_preserves_recorded_seeded_numpy_percentiles(seed, expected, budget):
    assert bootstrap_mean_ci([0.,1.,3.,9.,12.], n_boot=17, seed=seed,
                             max_resample_elements=budget) == expected


@pytest.mark.parametrize("seed, expected", [(0, (5.,3.,9.8)), (7, (5.,2.6,9.))])
def test_gym_convention_retains_its_recorded_generator_and_order_statistics(seed, expected):
    assert bootstrap_competence_ci([0.,1.,3.,9.,12.], n_boot=17, seed=seed) == expected


@pytest.mark.parametrize("n, budget, max_rows", [(7, 20, 2), (7, 1, 1)])
def test_bootstrap_allocations_respect_element_budget_with_one_resample_minimum(monkeypatch, n, budget, max_rows):
    original = np.random.default_rng
    shapes = []

    class Generator:
        def __init__(self, seed):
            self.rng = original(seed)
        def integers(self, *args, **kwargs):
            shapes.append(kwargs['size'])
            return self.rng.integers(*args, **kwargs)

    monkeypatch.setattr(np.random, 'default_rng', Generator)
    bootstrap_mean_ci(list(range(n)), n_boot=11, max_resample_elements=budget)
    assert sum(rows for rows, _ in shapes) == 11
    assert all(columns == n and 1 <= rows <= max_rows for rows, columns in shapes)
    assert len(shapes) > 1


def test_empty_and_constant_intervals_have_explicit_legacy_results():
    assert bootstrap_mean_ci([]) == (0.,0.)
    assert bootstrap_mean_ci([4.] * 5, alpha=.2) == (4.,4.)
    assert paired_diff_ci([]) == {'mean': 0., 'ci95': (0.,0.), 'lower_gt_zero': False}
    assert paired_diff_ci([-1.] * 5) == {'mean': -1., 'ci95': (-1.,-1.), 'lower_gt_zero': False}


def test_estimators_do_not_change_process_random_generators():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    bootstrap_mean_ci([0.,1.,2.], n_boot=11)
    bootstrap_competence_ci([0.,1.,2.], n_boot=11)
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]


@pytest.mark.parametrize("function", [bootstrap_mean_ci, bootstrap_competence_ci, paired_diff_ci])
@pytest.mark.parametrize("values", [[[1.]], [float('nan')], [float('inf')], ['bad'], None])
def test_nonfinite_or_malformed_observations_are_classified(function, values):
    with pytest.raises(InputError):
        function(values)


@pytest.mark.parametrize("function", [bootstrap_mean_ci, bootstrap_competence_ci])
@pytest.mark.parametrize("options", [{'n_boot': 0}, {'n_boot': True}, {'n_boot': 1.5},
                                    {'seed': -1}, {'seed': None}, {'alpha': 0.}, {'alpha': 1.},
                                    {'alpha': True}, {'alpha': '0.05'}, {'alpha': float('nan')}])
def test_invalid_resampling_requests_fail_even_for_empty_observations(function, options):
    with pytest.raises(ConfigurationError):
        function([], **options)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, None])
def test_invalid_allocation_budget_is_rejected_before_resampling(budget):
    with pytest.raises(ConfigurationError) as caught:
        bootstrap_mean_ci([1.], max_resample_elements=budget)
    assert caught.value.context['field'] == 'max_resample_elements'


@pytest.mark.parametrize("k,n", [(-1, 10), (1, -1), (True, 1), (1, 2.5)])
def test_wilson_rejects_invalid_integer_counts(k, n):
    with pytest.raises(ConfigurationError):
        wilson_interval(k, n)


def test_wilson_rejects_more_successes_than_observations():
    with pytest.raises(InputError):
        wilson_interval(2, 1)


@pytest.mark.parametrize("z", [0., -1., True, None, '1.96', float('inf'), float('nan')])
def test_wilson_rejects_invalid_critical_values(z):
    with pytest.raises(ConfigurationError):
        wilson_interval(0, 1, z=z)


def test_historical_estimator_imports_are_shared_aliases():
    from alignair.aa import evaluate as previous
    from alignair.train.gym.instrument.stats import bootstrap_ci
    assert previous.bootstrap_mean_ci is bootstrap_mean_ci
    assert previous.paired_diff_ci is paired_diff_ci
    assert previous.wilson_interval is wilson_interval
    assert bootstrap_ci is bootstrap_competence_ci
