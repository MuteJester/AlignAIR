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


# A V germline long enough for the matcher's 15-base anchors: fixed pseudo-random bases (seed 0).
_V = "".join(np.random.default_rng(0).choice(list("ACGT"), size=287))


def _v_alignment(read, start, end, indel_count):
    ref = _FakeRef({"V": _FakeGene(["V1"], [_V])})
    segs = Segments({"v": np.array([start])}, {"v": np.array([end])})
    calls = {"v": [GeneCall(("V1",), (0.9,))]}
    return align_germline([read], segs, calls, ref, indel_counts=np.array([indel_count]))["v"][0]


def _deleted(seq, *positions):
    """``seq`` without the bases at ``positions`` (germline coordinates)."""
    return "".join(base for i, base in enumerate(seq) if i not in positions)


def _columns(read, a):
    """``(mismatches, inserted, deleted)`` of alignment ``a`` of ``read`` on ``_V``."""
    import re
    mismatches = inserted = deleted = 0
    r, g = a.seq_start, a.germ_start
    for n, op in re.findall(r"(\d+)([MID])", a.cigar):
        n = int(n)
        if op == "M":
            mismatches += sum(x != y for x, y in zip(read[r:r + n], _V[g:g + n]))
            r, g = r + n, g + n
        elif op == "I":
            inserted, r = inserted + n, r + n
        else:
            deleted, g = deleted + n, g + n
    return mismatches, inserted, deleted


@pytest.mark.parametrize("positions", [(16, 54), (12, 70), (4, 5)])
def test_a_whole_v_with_a_deletion_near_its_start_is_placed_from_the_germline_start(positions):
    """A read showing the whole V, with a deletion in its first bases: the start derived from the end anchor and the
    shortened length lands past the germline's first base, and the CIGAR used to open with read bases on no germline
    (``2I14M1D...`` at germline start 2). The start is settled where the alignment fits best: the germline start,
    every read base on an equal germline base and the deleted bases as deletions (placed anywhere along a run of
    equal bases)."""
    read = _deleted(_V, *positions)
    a = _v_alignment(read, 0, len(read), indel_count=0.4)
    assert (a.seq_start, a.seq_end, a.germ_start, a.germ_end) == (0, len(read), 0, len(_V))
    assert a.cigar.split("M")[0].isdigit()              # opens with read bases on germline bases
    assert _columns(read, a) == (0, 0, len(positions))


def test_settling_a_start_never_moves_it_onto_read_bases_that_do_not_match_the_germline():
    """Two read bases before the V's first observed base (germline 40) that differ from the germline bases before
    it: from the right start, the settled start stays there (moving onto them gains nothing)."""
    from alignair.predict.heuristic_matcher import derive_alignment, settle_start
    flip = {"A": "C", "C": "A", "G": "T", "T": "G"}
    read = flip[_V[38]] + flip[_V[39]] + _deleted(_V[40:], 10)
    cigar = derive_alignment(read, _V[40:], 0.4).cigar
    assert settle_start(read, _V, 40, len(_V), 0.4, cigar) == (40, cigar)


def test_a_truncated_v_with_a_deletion_near_its_first_base_keeps_its_start():
    read = _deleted(_V, 50)[40:]                           # germline 40.. with germline base 50 deleted
    a = _v_alignment(read, 0, len(read), indel_count=0.4)
    assert (a.germ_start, a.germ_end, a.cigar) == (40, len(_V), "10M1D236M")


@pytest.mark.parametrize("mutated", [(2, 3, 4), (1, 2)])
def test_a_whole_v_whose_first_bases_are_mutated_keeps_its_start(mutated):
    """Somatic mutations in a read's first bases are mismatches, not a reason to move the start past them: the
    start stays on the germline's first base, the mutated bases aligned as mismatches."""
    flip = {"A": "C", "C": "A", "G": "T", "T": "G"}
    read = "".join(flip[b] if i in mutated else b for i, b in enumerate(_V))
    a = _v_alignment(read, 0, len(read), indel_count=0.4)
    assert (a.germ_start, a.cigar) == (0, f"{len(_V)}M")


@pytest.mark.parametrize("read, start, end, cigar", [(_V[7:10], 6, 293, "278D1M6D2M"), (_V[100:102], 99, 103, "1M2D1M")])
def test_settling_leaves_a_degenerate_placement_as_placed(read, start, end, cigar):
    """Three read bases over a 287-base germline window (``278D1M6D2M``), or a segment shorter than the matcher's
    anchor: no nearby start repairs such a placement, so none is searched and the matcher's stands."""
    from alignair.predict.heuristic_matcher import settle_start
    assert settle_start(read, _V, start, end, 0.0, cigar) == (start, cigar)


_FLIP = {"A": "C", "C": "A", "G": "T", "T": "G"}


def _unlike(seq):
    """Bases each unlike ``seq``'s (an N region that matches no germline base)."""
    return "".join(_FLIP[b] for b in seq)


def test_a_v_end_the_model_stopped_short_of_a_deletion_is_extended_past_it():
    """The whole V with germline base 281 deleted, then N bases: the model's V segment stops at read 279, two bases
    short of the deletion. The read bases after it follow the germline past the deletion, so the end moves to the
    germline's last base (read 286, germline 287), the deletion inside the alignment."""
    read = _deleted(_V, 281) + _unlike(_V[:10])
    a = _v_alignment(read, 0, 279, indel_count=0.6)
    assert (a.seq_start, a.seq_end, a.germ_start, a.germ_end) == (0, 286, 0, len(_V))
    assert _columns(read, a) == (0, 0, 1)


def test_a_v_end_followed_by_bases_unlike_the_germline_stays():
    read = _V[:270] + _unlike(_V[270:290])
    a = _v_alignment(read, 0, 270, indel_count=0.2)
    assert (a.seq_end, a.germ_end, a.cigar) == (270, 270, "270M")


def test_a_v_end_past_its_last_germline_base_is_trimmed_back():
    """The model's V segment runs three bases into an N region unlike the germline: the end comes back to the V's
    last base."""
    read = _V[:280] + _unlike(_V[280:290])
    a = _v_alignment(read, 0, 283, indel_count=0.2)
    assert (a.seq_end, a.germ_end, a.cigar) == (280, 280, "280M")


def test_a_v_end_is_not_moved_past_its_limit():
    """The read position the V may reach (the next segment's start) bounds an extension."""
    from alignair.predict.heuristic_matcher import derive_alignment, settle_end
    read = _deleted(_V, 281) + _unlike(_V[:10])
    cigar = derive_alignment(read[:279], _V[:279], 0.6).cigar
    seq_end, germ_end = settle_end(read, _V, 0, 279, 0, 279, cigar, 282)
    assert seq_end <= 282
