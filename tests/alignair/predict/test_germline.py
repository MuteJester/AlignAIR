"""Tests for germline alignment (reuse align/ WFA/parasail; Short-D sentinel)."""
import numpy as np

from alignair.predict.germline import align_germline
from alignair.predict.state import GeneCall, Segments


class _FakeGene:
    def __init__(self, names, seqs):
        self.names, self.sequences = names, seqs
        self.index = {n: i for i, n in enumerate(names)}


class _FakeRef:
    def __init__(self, genes):
        self._g = genes

    def gene(self, g):
        return self._g[g]


def test_exact_match_gives_full_germline_coords_and_cigar():
    germ = "ACGTACGTACGT"
    ref = _FakeRef({"V": _FakeGene(["V1"], [germ])})
    sequences = ["TTT" + germ + "GGG"]                  # V segment at read [3:15]
    segs = Segments({"v": np.array([3])}, {"v": np.array([15])})
    calls = {"v": [GeneCall(("V1",), (0.9,))]}
    a = align_germline(sequences, segs, calls, ref)["v"][0]
    assert a.allele == "V1"
    assert a.germ_start == 0 and a.germ_end == len(germ)
    assert a.seq_start == 3 and a.seq_end == 15
    assert "M" in a.cigar                               # exact match -> match ops


def test_short_d_empty_reference_sentinel():
    ref = _FakeRef({"D": _FakeGene(["Short-D"], [""])})
    segs = Segments({"d": np.array([2])}, {"d": np.array([5])})
    calls = {"d": [GeneCall(("Short-D",), (0.5,))]}
    a = align_germline(["ACGTACGT"], segs, calls, ref)["d"][0]
    assert a.germ_start == 0 and a.germ_end == 0 and a.cigar == ""


def test_empty_call_yields_none():
    ref = _FakeRef({"V": _FakeGene(["V1"], ["ACGT"])})
    segs = Segments({"v": np.array([0])}, {"v": np.array([4])})
    calls = {"v": [GeneCall((), ())]}
    assert align_germline(["ACGT"], segs, calls, ref)["v"][0] is None
