"""Tests for genotype-based likelihood correction."""
import numpy as np

from alignair.genotype.constraint import adjust_for_genotype
from alignair.predict.state import Predictions


class _FakeGene:
    def __init__(self, names):
        self.names = names


class _FakeRef:
    def __init__(self, genes):
        self._g = genes

    def gene(self, g):
        return self._g[g]


def test_drops_non_genotype_and_redistributes():
    ref = _FakeRef({"V": _FakeGene(["V1", "V2", "V3"])})
    preds = Predictions(allele={"v": np.array([[0.6, 0.3, 0.1]])}, start={}, end={},
                        mutation_rate=np.array([0.0]), indel_count=np.array([0.0]),
                        productive=np.array([True]))
    adjust_for_genotype(preds, {"v": {"V1", "V2"}}, ref)
    row = preds.allele["v"][0]
    assert row[2] == 0.0                         # V3 dropped
    assert row[0] > 0.6 and row[1] > 0.3          # 0.1 mass redistributed proportionally
    assert row.max() <= 1.0
