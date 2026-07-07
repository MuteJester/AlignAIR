"""Tests for allele selection (MaxLikelihoodPercentageThreshold, the production selector)."""
import numpy as np

from alignair.predict.threshold import max_likelihood_percentage, select_alleles
from alignair.predict.state import GeneCall


def test_relative_to_max_filter_sort_and_cap():
    # p_i >= pct*max(p): max=0.9, pct=0.1 -> bar=0.09. keep {0.9,0.5,0.3,0.1}, drop 0.05; cap 3.
    p = np.array([0.9, 0.05, 0.5, 0.1, 0.3])
    idx, lk = max_likelihood_percentage(p, pct=0.1, cap=3)
    assert list(idx) == [0, 2, 4]                     # sorted desc by prob, capped at 3
    assert np.allclose(lk, [0.9, 0.5, 0.3])


def test_below_bar_dropped():
    p = np.array([0.8, 0.05, 0.02])                   # bar = 0.08 -> only index 0
    idx, lk = max_likelihood_percentage(p, pct=0.1, cap=3)
    assert list(idx) == [0]


def test_select_alleles_maps_indices_to_names():
    names = {"v": ["V1", "V2", "V3"], "j": ["J1", "J2"]}
    preds = {"v": np.array([[0.9, 0.5, 0.05]]), "j": np.array([[0.2, 0.8]])}
    calls = select_alleles(preds, names, pct=0.1, cap=3)
    assert isinstance(calls["v"][0], GeneCall)
    assert calls["v"][0].names == ("V1", "V2")        # V3 (0.05 < 0.09 bar) dropped
    assert calls["j"][0].names == ("J2", "J1")        # sorted desc
