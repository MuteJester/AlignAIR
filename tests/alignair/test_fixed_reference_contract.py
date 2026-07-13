"""Black-box acceptance tests for the fixed-reference model contract (docs/architecture/model_contract.md,
P0-1): a novel allele absent from the trained catalog is REJECTED with a clear NovelAlleleUnsupportedError
rather than silently dropped or mis-indexed; a donor subset of the trained catalog is accepted."""
from types import SimpleNamespace

import numpy as np
import pytest

from alignair.genotype.constraint import (NovelAlleleUnsupportedError, adjust_for_genotype,
                                           genotype_allowed_mask, load_genotype)
from alignair.predict.state import Predictions


class _Ref:
    def __init__(self, names):
        self._n = names

    def gene(self, g):
        return SimpleNamespace(names=self._n)


def _preds(v_probs):
    return Predictions(start={}, end={}, allele={"v": np.array(v_probs, dtype=np.float64)},
                       mutation_rate=np.zeros(len(v_probs)), indel_count=np.zeros(len(v_probs)),
                       productive=np.zeros(len(v_probs)))


def test_novel_allele_is_rejected_not_dropped():
    """The defining fixed-reference guarantee: a novel allele cannot become callable at inference."""
    ref = _Ref(["IGHV1-2*02", "IGHV3-23*01"])
    with pytest.raises(NovelAlleleUnsupportedError):
        genotype_allowed_mask({"v": {"IGHV_NOVEL*01"}}, ref)
    assert issubclass(NovelAlleleUnsupportedError, ValueError)   # back-compat for except ValueError


def test_novel_allele_rejected_through_load_genotype(tmp_path):
    import json
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"v": ["IGHV1-2*02", "IGHV_NOVEL*01"]}))
    ref = _Ref(["IGHV1-2*02", "IGHV3-23*01"])
    with pytest.raises(NovelAlleleUnsupportedError, match="not in the model reference|cannot call"):
        load_genotype(str(p), reference=ref, drop_unknown=False)


def test_donor_subset_of_trained_catalog_is_accepted():
    """A genotype that is a subset of the trained catalog is honored (donor subsetting is supported)."""
    ref = _Ref(["IGHV1-2*02", "IGHV3-23*01", "IGHV4-34*01"])
    out = adjust_for_genotype(_preds([[0.6, 0.5, 0.4]]), {"v": {"IGHV1-2*02", "IGHV4-34*01"}},
                              ref, method="mask")
    assert out.allele["v"][0][1] == 0.0                          # the out-of-genotype allele is masked
    mask = genotype_allowed_mask({"v": {"IGHV1-2*02", "IGHV4-34*01"}}, ref)["v"]
    assert list(mask) == [True, False, True]
