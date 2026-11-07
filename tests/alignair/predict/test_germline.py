"""Tests for germline alignment (the heuristic matcher; Short-D sentinel)."""
import pytest
import numpy as np

from alignair.predict.germline import align_germline, realign_v
from alignair.predict.state import GeneCall, Segments

pytestmark = pytest.mark.component


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


# IGHD3-10*01
_D = "GTATTACTATGGTTCGGGGAGTTATTATAAC"


def _d_alignment(read, start, end, indel_count):
    ref = _FakeRef({"D": _FakeGene(["D1"], [_D])})
    segs = Segments({"d": np.array([start])}, {"d": np.array([end])})
    calls = {"d": [GeneCall(("D1",), (0.9,))]}
    return align_germline([read], segs, calls, ref, indel_counts=np.array([indel_count]))["d"][0]


def _matches(read, a):
    return sum(x == y for x, y in zip(read[a.seq_start:a.seq_end], _D[a.germ_start:a.germ_end]))


def test_a_d_holding_an_insertion_is_placed_without_a_gap():
    """IgBLAST and IMGT align a D without gaps: a D germline span always equals its read span. A D whose read bases
    hold an inserted base is placed along its germline at the gapless offset that matches most of them."""
    segment = _D[5:15] + "A" + _D[15:25]
    read = "CCCC" + segment + "GGGG"
    a = _d_alignment(read, 4, 4 + len(segment), indel_count=1.0)
    assert a.cigar == f"{a.seq_end - a.seq_start}M"
    assert a.germ_end - a.germ_start == a.seq_end - a.seq_start
    assert _matches(read, a) >= 10                      # at least the larger of the insertion's two sides


def test_a_d_hanging_off_its_germline_is_cut_to_the_bases_the_germline_covers():
    """Read bases before the germline's first base (an N1 base the segment took) leave the D: the gapless placement
    covers only the bases the germline holds."""
    read = "CCCC" + "TCC" + _D[:12] + "GGGG"
    a = _d_alignment(read, 4, 4 + 15, indel_count=1.0)
    assert (a.seq_start, a.seq_end, a.germ_start, a.germ_end, a.cigar) == (7, 19, 0, 12, "12M")


def test_a_d_without_read_bases_spans_no_germline():
    """A D segment of no read bases is placed on no germline base (not on the whole germline as deletions)."""
    a = _d_alignment("CCCC" + "GGGG", 4, 4, indel_count=0.0)
    assert (a.seq_start, a.seq_end, a.germ_start, a.germ_end, a.cigar) == (4, 4, 0, 0, "")


def test_realign_v_aligns_a_reads_new_v_as_a_called_allele_and_keeps_the_others():
    a = "ACGTACGTTGCAACGTAGGCTTACGATCGA"
    b = a[:12] + "G" + a[13:]                            # a's sibling, one SNP at germline position 12
    ref = _FakeRef({"V": _FakeGene(["A", "B"], [a, b])})
    sequences = ["TT" + b + "CC", "TT" + a + "CC"]
    segs = Segments({"v": np.array([2, 2])}, {"v": np.array([32, 32])})
    first = align_germline(sequences, segs, {"v": [GeneCall(("A",), (0.9,))] * 2}, ref)["v"]
    as_b = align_germline(sequences[:1], Segments({"v": np.array([2])}, {"v": np.array([32])}),
                          {"v": [GeneCall(("B",), (0.9,))]}, ref)["v"][0]
    out = realign_v(sequences, segs, first, ["B", None], ref)
    assert out[0] == as_b and out[0].allele == "B"
    assert out[1] is first[1]


def test_realign_v_keeps_the_current_alignment_when_the_new_allele_matches_fewer_read_bases():
    a = "ACGTACGTTGCAACGTAGGCTTACGATCGA"
    c = a[:5] + "TTT" + a[8:20] + "CCCC" + a[24:]          # differs from the read in 7 places
    ref = _FakeRef({"V": _FakeGene(["A", "C"], [a, c])})
    sequences = ["TT" + a[:12] + "G" + a[13:] + "CC"]      # a with one mismatch
    segs = Segments({"v": np.array([2])}, {"v": np.array([32])})
    first = align_germline(sequences, segs, {"v": [GeneCall(("A",), (0.9,))]}, ref)["v"]
    assert realign_v(sequences, segs, first, ["C"], ref)[0] is first[0]
