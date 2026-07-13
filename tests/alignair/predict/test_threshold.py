"""Tests for allele-set selection. The default is the derived, calibration-free ``absolute`` rule
(p >= 0.5, the BCE-calibrated decision boundary); ``largest_gap`` is a parameter-free alternative;
``max_likelihood_percentage`` is the legacy relative-to-max rule kept for comparison."""
import numpy as np

from alignair.predict.threshold import (absolute_threshold, largest_gap,
                                         max_likelihood_percentage, select_alleles)
from alignair.predict.state import GeneCall


def test_absolute_keeps_calibrated_members_only():
    # BCE-calibrated: keep p >= 0.5. members {0.9, 0.6}; 0.4/0.1 dropped.
    p = np.array([0.9, 0.4, 0.6, 0.1])
    idx, lk = absolute_threshold(p, thr=0.5, cap=3)
    assert list(idx) == [0, 2] and np.allclose(lk, [0.9, 0.6])


def test_absolute_never_empty_falls_back_to_top1():
    p = np.array([0.3, 0.1, 0.2])                      # nothing >= 0.5 -> keep the argmax
    idx, lk = absolute_threshold(p, thr=0.5, cap=3)
    assert list(idx) == [0] and np.allclose(lk, [0.3])


def test_largest_gap_cuts_at_biggest_drop():
    p = np.array([0.9, 0.85, 0.1, 0.05])              # biggest drop 0.85->0.1 -> keep top-2
    idx, _ = largest_gap(p, cap=3)
    assert set(idx.tolist()) == {0, 1}
    p2 = np.array([0.9, 0.1, 0.08])                    # biggest drop 0.9->0.1 -> keep top-1
    assert list(largest_gap(p2, cap=3)[0]) == [0]


def test_legacy_percentage_still_available():
    p = np.array([0.9, 0.05, 0.5, 0.1, 0.3])           # bar=0.09; keep {0.9,0.5,0.3}, cap 3
    idx, lk = max_likelihood_percentage(p, pct=0.1, cap=3)
    assert list(idx) == [0, 2, 4] and np.allclose(lk, [0.9, 0.5, 0.3])


def test_select_alleles_default_is_absolute():
    names = {"v": ["V1", "V2", "V3"], "j": ["J1", "J2"]}
    preds = {"v": np.array([[0.9, 0.6, 0.2]]), "j": np.array([[0.2, 0.8]])}
    calls = select_alleles(preds, names)               # default selector="absolute", threshold=0.5
    assert isinstance(calls["v"][0], GeneCall)
    assert calls["v"][0].names == ("V1", "V2")         # V3 (0.2 < 0.5) dropped, no calibration
    assert calls["j"][0].names == ("J2",)              # only J2 >= 0.5


def test_select_alleles_supports_largest_gap():
    names = {"v": ["V1", "V2", "V3"]}
    preds = {"v": np.array([[0.9, 0.85, 0.1]])}
    calls = select_alleles(preds, names, selector="largest_gap")
    assert set(calls["v"][0].names) == {"V1", "V2"}


# --- P0-5: an allowed-index set makes constrained calls always members of the set ----------------

def test_allowed_set_restricts_call_even_when_disallowed_has_max_prob():
    names = {"v": ["V1", "V2", "V3"]}
    preds = {"v": np.array([[0.99, 0.30, 0.20]])}        # V1 (disallowed) has the max probability
    allowed = {"v": np.array([False, True, True])}       # only V2/V3 allowed
    calls = select_alleles(preds, names, allowed=allowed)
    assert "V1" not in calls["v"][0].names               # never call an out-of-genotype allele
    assert calls["v"][0].names[0] == "V2"                # best allowed allele (top-1 among allowed)


def test_allowed_set_never_calls_index0_when_all_probs_zero():
    names = {"v": ["V1", "V2", "V3"]}
    preds = {"v": np.zeros((1, 3))}                       # degenerate: every probability exactly 0
    allowed = {"v": np.array([False, False, True])}       # only V3 allowed
    calls = select_alleles(preds, names, allowed=allowed)
    assert calls["v"][0].names == ("V3",)                # not the default argmax index 0 (disallowed)
