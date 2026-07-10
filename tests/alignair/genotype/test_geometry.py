"""Genotype task 1: allele prototype geometry + asymmetric leakage model."""
import numpy as np

from alignair.genotype.geometry import (LeakageModel, allele_prototypes, prototype_cosine,
                                        residual_support)


def test_prototypes_shape_from_model():
    from alignair.core import AlignAIR
    from alignair.core.config import AlignAIRConfig
    cfg = AlignAIRConfig(v_allele_count=4, j_allele_count=2, d_allele_count=1, has_d=True,
                         max_seq_length=128)
    W, b = allele_prototypes(AlignAIR(cfg), "v")
    assert W.shape[0] == 4 and W.ndim == 2 and b.shape == (4,)     # one prototype row per allele


def test_cosine_siblings_closer_than_random():
    W = np.array([[1, 0, 0.1], [1, 0, 0.0], [0, 1, 0.1], [0, 1, 0.0]], dtype=float)  # (0,1),(2,3) siblings
    C = prototype_cosine(W)
    sib = (C[0, 1] + C[2, 3]) / 2
    rnd = (C[0, 2] + C[0, 3] + C[1, 2] + C[1, 3]) / 4
    assert sib > rnd


def test_leakage_is_asymmetric_via_attractiveness():
    W = np.array([[1, 0, 0.1], [1, 0, 0.0], [0, 1, 0.0], [0, 1, 0.0]], dtype=float)
    L = LeakageModel.fit(W, biases=np.array([0.0, 3.0, 0.0, 0.0]))   # allele 1 more 'attractive'
    assert L.predict(0, 1) > L.predict(1, 0)                          # more leakage INTO 1 than into 0


def test_residual_subtracts_leakage_from_all_present_neighbors():
    W = np.array([[1, 0, 0.0], [1, 0, 0.0], [0, 1, 0.0]], dtype=float)   # 0,1 siblings; 2 distant
    L = LeakageModel.fit(W)
    residual = residual_support({0: 1.0, 1: 0.3, 2: 0.0}, L, present={0})
    assert residual[1] < 0.3                                          # sibling reduced by leakage from 0
    assert residual[0] == 1.0                                         # dominant unaffected


def test_calibration_reduces_leakage_when_homozygous_shows_little():
    W = np.array([[1, 0, 0.0], [1, 0, 0.0]], dtype=float)
    seen = {0: {1: 0.02}}                                             # homozygous-for-0: only 2% leaks to 1
    calib = LeakageModel.fit(W, homozygous_leakage=seen)
    raw = LeakageModel.fit(W)
    assert calib.predict(0, 1) <= raw.predict(0, 1)                   # calibrated to the observed low rate
