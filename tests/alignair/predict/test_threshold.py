"""Allele-set selection: the alleles scoring at least the threshold (0.5 by default), best first, capped, and never
empty (the best allele stays when none reaches it)."""
import pytest
import numpy as np

from alignair.predict.threshold import select_alleles
from alignair.predict.state import GeneCall

pytestmark = pytest.mark.component


def test_the_set_keeps_the_alleles_at_or_above_the_threshold_best_first():
    calls = select_alleles({"v": np.array([[0.6, 0.4, 0.9, 0.1]])}, {"v": ["V1", "V2", "V3", "V4"]})
    assert calls["v"][0].names == ("V3", "V1") and np.allclose(calls["v"][0].likelihoods, [0.9, 0.6])


def test_the_set_is_never_empty_and_keeps_the_best_allele():
    calls = select_alleles({"v": np.array([[0.3, 0.1, 0.2]])}, {"v": ["V1", "V2", "V3"]})
    assert calls["v"][0].names == ("V1",) and np.allclose(calls["v"][0].likelihoods, [0.3])


def test_the_set_is_capped():
    calls = select_alleles({"v": np.array([[0.9, 0.8, 0.7, 0.6]])}, {"v": ["V1", "V2", "V3", "V4"]}, cap=2)
    assert calls["v"][0].names == ("V1", "V2")


def test_select_alleles_calls_every_gene_with_the_default_threshold():
    names = {"v": ["V1", "V2", "V3"], "j": ["J1", "J2"]}
    preds = {"v": np.array([[0.9, 0.6, 0.2]]), "j": np.array([[0.2, 0.8]])}
    calls = select_alleles(preds, names)
    assert isinstance(calls["v"][0], GeneCall)
    assert calls["v"][0].names == ("V1", "V2")
    assert calls["j"][0].names == ("J2",)


def test_alleles_tied_at_a_score_keep_the_banks_order():
    """A saturated head gives several alleles the same float32 score; tied alleles are kept in the bank's order, so
    the set does not depend on numpy's sort, which orders ties differently across versions and CPUs. The test chip's
    J scores of one golden read (four alleles tied behind the best, the cap reached among them), and a best allele
    ahead of nineteen tied ones."""
    scores = np.array([[0.9999982, 0.9999989, 1.0, 0.9999988, 0.9999989, 0.9999989, 0.9999989]], dtype=np.float32)
    names = ["J1", "J2", "J3", "J4", "J5", "J6", "J7"]
    assert select_alleles({"j": scores}, {"j": names})["j"][0].names == ("J3", "J2", "J5")
    many = np.full((1, 20), 0.75)
    many[0, 10] = 0.9
    assert select_alleles({"v": many}, {"v": [f"V{k}" for k in range(20)]})["v"][0].names == ("V10", "V0", "V1")
