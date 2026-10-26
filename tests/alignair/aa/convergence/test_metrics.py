import pytest
import numpy as np
from alignair.aa.convergence.metrics import macro_accuracy, per_frequency_bins, softmax_nll, brier_score

pytestmark = pytest.mark.component

def test_macro_downweights_common_class():
    # class 0 has 8 examples all correct; class 1 has 2 examples all wrong -> micro 0.8, macro 0.5
    yt = np.array([0]*8 + [1]*2); yp = np.array([0]*8 + [0]*2)
    assert abs(macro_accuracy(yt, yp) - 0.5) < 1e-9

def test_frequency_bins():
    yt = np.array([0,0,1]); yp = np.array([0,1,1]); freq = np.array([0.9,0.9,0.01])
    out = per_frequency_bins(yt, yp, freq, edges=(0.0,0.05,1.0))
    assert out["[0.0,0.05)"]["acc"] == 1.0 and out["[0.05,1.0]"]["acc"] == 0.5

def test_nll_and_brier_perfect_and_uniform():
    probs = np.array([[1.0,0.0],[0.0,1.0]]); yt = np.array([0,1])
    assert softmax_nll(np.clip(probs,1e-12,1), yt) < 1e-6
    assert brier_score(probs, yt) < 1e-9
    unif = np.array([[0.5,0.5],[0.5,0.5]])
    assert abs(brier_score(unif, yt) - 0.5) < 1e-9   # sum (0.5-*)^2 = 0.5 per row
