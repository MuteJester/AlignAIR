import numpy as np
from alignair.postprocessing.allele_selector import max_likelihood_threshold, select_alleles


def test_threshold_selects_above_fraction_of_max():
    probs = np.array([0.9, 0.5, 0.1, 0.05])
    idx, lik = max_likelihood_threshold(probs, percentage=0.21, cap=3)
    # threshold = 0.9*0.21 = 0.189 -> indices 0 (0.9) and 1 (0.5)
    assert list(idx) == [0, 1]
    assert np.allclose(lik, [0.9, 0.5])


def test_threshold_cap():
    probs = np.array([0.9, 0.85, 0.8, 0.75, 0.7])
    idx, _ = max_likelihood_threshold(probs, percentage=0.5, cap=3)
    assert len(idx) == 3  # capped
    assert list(idx) == [0, 1, 2]  # top 3 by likelihood


def test_select_alleles_maps_names():
    probs = np.array([[0.9, 0.1, 0.05], [0.1, 0.8, 0.7]])
    index_to_allele = {0: "A*01", 1: "B*01", 2: "C*01"}
    out = select_alleles(probs, index_to_allele, percentage=0.21, cap=3)
    assert out[0][0] == ["A*01"]
    assert set(out[1][0]) == {"B*01", "C*01"}
