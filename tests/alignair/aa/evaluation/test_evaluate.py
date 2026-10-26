import pytest
from alignair.aa.evaluate import (
    bootstrap_mean_ci, constant_median_mae, majority_class_baseline, paired_diff_ci,
    wilson_interval,
)

pytestmark = pytest.mark.component


def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = wilson_interval(80, 100)
    assert lo < 0.8 < hi and 0.0 <= lo and hi <= 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)
    tight = wilson_interval(8000, 10000)
    wide = wilson_interval(8, 10)
    assert (tight[1] - tight[0]) < (wide[1] - wide[0])       # more data -> tighter


def test_majority_class_baseline_is_the_best_constant_predictor():
    assert majority_class_baseline([1, 1, 1, 2]) == 0.75
    assert majority_class_baseline([]) == 0.0


def test_constant_median_mae_beats_or_matches_any_other_constant():
    vals = [1.0, 2.0, 3.0, 10.0]
    med = constant_median_mae(vals)
    for c in (0.0, 1.5, 4.0, 11.0):
        assert med <= sum(abs(v - c) for v in vals) / len(vals) + 1e-9


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
