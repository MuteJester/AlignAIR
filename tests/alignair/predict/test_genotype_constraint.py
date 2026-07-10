"""Genotype-constrained inference: mask (accuracy) + renormalize (confidence)."""
import numpy as np

from alignair.predict.genotype import adjust_for_genotype, load_genotype
from alignair.predict.state import Predictions


def _preds(v_probs):
    return Predictions(start={}, end={}, allele={"v": np.array(v_probs, dtype=np.float64)},
                       mutation_rate=np.zeros(len(v_probs)), indel_count=np.zeros(len(v_probs)),
                       productive=np.zeros(len(v_probs)), orientation=None, chain_type=None)


class _Gene:
    def __init__(self, names):
        self.names = names


class _Ref:
    def __init__(self, names):
        self._n = names

    def gene(self, g):
        return _Gene(self._n)


def test_mask_zeros_out_of_genotype_and_keeps_in_genotype():
    ref = _Ref(["V1", "V2", "V3", "V4"])
    out = adjust_for_genotype(_preds([[0.9, 0.8, 0.7, 0.95]]), {"v": {"V1", "V2", "V3"}}, ref, method="mask")
    row = out.allele["v"][0]
    assert row[3] == 0.0                                  # V4 (out of genotype) zeroed
    assert list(row[:3]) == [0.9, 0.8, 0.7]              # in-genotype probs UNCHANGED (mask only)


def test_renormalize_conditions_and_sharpens_confidence():
    ref = _Ref(["V1", "V2", "V3"])
    # V3 (a distractor) held 0.5 of the mass; excluding + renormalizing lifts the true V1 0.4 -> 0.8
    out = adjust_for_genotype(_preds([[0.4, 0.1, 0.5]]), {"v": {"V1", "V2"}}, ref, method="renormalize")
    row = out.allele["v"][0]
    assert row[2] == 0.0                                  # V3 excluded
    assert abs(row.sum() - 1.0) < 1e-9                    # posterior sums to 1 over the allowed set
    assert abs(row[0] - 0.8) < 1e-9 and abs(row[1] - 0.2) < 1e-9   # confidence sharpened upward


def test_softmax_sharpens_a_confident_top_over_the_allowed_set():
    ref = _Ref(["V1", "V2", "V3"])
    # logit(0.9)=+2.2, logit(0.1)=-2.2: softmax over {V1,V2} concentrates on V1 -> ~0.99 (> 0.9)
    out = adjust_for_genotype(_preds([[0.9, 0.1, 0.1]]), {"v": {"V1", "V2"}}, ref, method="softmax")
    row = out.allele["v"][0]
    assert row[2] == 0.0 and abs(row.sum() - 1.0) < 1e-9
    assert row[0] > 0.95                                  # confident top sharpened upward


def test_renormalize_promotes_borderline_true_allele_over_threshold():
    ref = _Ref(["V1", "V2", "V3"])
    # true allele V1 at 0.4 (below 0.5) competing with two distractors; masking distractors + renorm
    out = adjust_for_genotype(_preds([[0.4, 0.35, 0.3]]), {"v": {"V1"}}, ref, method="renormalize")
    assert out.allele["v"][0][0] == 1.0                  # sole allowed allele -> posterior 1.0 (recovered)


def test_load_genotype_json_and_yaml(tmp_path):
    import json
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"v": ["IGHV1-2*01", "IGHV3-23*01"], "j": ["IGHJ4*02"]}))
    g = load_genotype(str(p))
    assert g["v"] == {"IGHV1-2*01", "IGHV3-23*01"} and g["j"] == {"IGHJ4*02"}
    y = tmp_path / "g.yaml"
    y.write_text("v:\n  - IGHV1-2*01\nd:\n  - IGHD3-3*01\n")
    gy = load_genotype(str(y))
    assert gy["v"] == {"IGHV1-2*01"} and gy["d"] == {"IGHD3-3*01"}


def test_load_genotype_validates_against_reference(tmp_path):
    import json
    import pytest
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"v": ["IGHV1-2*01", "NOT-A-REAL-ALLELE"]}))
    ref = _Ref(["IGHV1-2*01", "IGHV3-23*01"])
    g, unknown = load_genotype(str(p), reference=ref, drop_unknown=True)
    assert g["v"] == {"IGHV1-2*01"} and "NOT-A-REAL-ALLELE" in unknown["v"]
    with pytest.raises(ValueError, match="unknown"):
        load_genotype(str(p), reference=ref, drop_unknown=False)
