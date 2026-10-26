import numpy as np
import pytest

from alignair.aa.convergence.paired import (
    paired_bootstrap_diff_ci,
    mcnemar,
    class_stratified_bootstrap_macro_diff,
)

pytestmark = pytest.mark.component


def test_paired_bootstrap_diff_ci_point_estimate_and_bounds():
    a = np.array([True, True, False, False])
    b = np.array([True, True, True, False])
    out = paired_bootstrap_diff_ci(a, b, n_boot=2000, seed=0, alpha=0.05)
    assert abs(out["diff"] - 0.25) < 1e-9
    assert out["lo"] <= 0.25 <= out["hi"]
    assert out["n"] == 4


def test_paired_bootstrap_diff_ci_length_mismatch_raises():
    a = np.array([True, False, True])
    b = np.array([True, False])
    with pytest.raises(ValueError):
        paired_bootstrap_diff_ci(a, b)


def test_paired_bootstrap_diff_ci_empty_raises():
    a = np.array([], dtype=bool)
    b = np.array([], dtype=bool)
    with pytest.raises(ValueError):
        paired_bootstrap_diff_ci(a, b)


def test_mcnemar_discordant_counts_and_pvalue():
    a = np.array([True, True, False, False, False])
    b = np.array([True, False, True, True, False])
    out = mcnemar(a, b)
    assert out["b_gain"] == 2  # idx 2,3: a False -> b True
    assert out["b_loss"] == 1  # idx 1: a True -> b False
    assert out["net"] == 1
    assert 0 < out["p_value"] <= 1


def test_mcnemar_no_discordant_pairs():
    a = np.array([True, False, True, False])
    b = np.array([True, False, True, False])
    out = mcnemar(a, b)
    assert out["b_gain"] == 0
    assert out["b_loss"] == 0
    assert out["p_value"] == 1.0
    assert out["net"] == 0


def test_class_stratified_bootstrap_macro_diff_point_estimate():
    y_true = np.array([0, 0, 1, 1])
    a_correct = np.array([True, True, False, False])   # class0 acc 1.0, class1 acc 0.0 -> macro_a 0.5
    b_correct = np.array([True, True, True, False])    # class0 acc 1.0, class1 acc 0.5 -> macro_b 0.75
    out = class_stratified_bootstrap_macro_diff(y_true, a_correct, b_correct, n_boot=2000, seed=0, alpha=0.05)
    assert abs(out["diff"] - 0.25) < 1e-9
    assert out["lo"] <= out["diff"] <= out["hi"]
