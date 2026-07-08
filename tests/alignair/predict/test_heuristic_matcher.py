"""Behaviour tests for the heuristic germline matcher (clean port of TF HeuristicReferenceMatcher).

The exhaustive identical-results check vs the TF logic was run over the whole benchmark
(7920 matches, V/D/J x fractional indel counts) at port time; these lock in the key behaviours.
"""
from alignair.predict.heuristic_matcher import GermlineMatch, HeuristicGermlineMatcher

_G = "ACGTACGTAC" * 4          # a 40 bp germline


def _matcher(**kw):
    return HeuristicGermlineMatcher({"V1": _G}, **kw)


def test_exact_clean_segment_quick_exits_to_full_germline():
    m = _matcher().match_one(_G, 0, 40, "V1", indel_count=0)
    assert m == GermlineMatch(seq_start=0, seq_end=40, ref_start=0, ref_end=40)


def test_5prime_truncated_segment_anchors_to_germline_suffix():
    # read carries only germline[10:] (a 5'-truncated V) with flanking noise
    read = "N" * 5 + _G[10:] + "N" * 3
    m = _matcher().match_one(read, 5, 5 + 30, "V1", indel_count=0)
    assert m.ref_start == 10 and m.ref_end == 40          # anchored on the conserved 3' end


def test_symmetric_overhang_is_trimmed_before_matching():
    read = "TT" + _G + "TT"                                # 44 bp: 2 bp overhang each side
    m = _matcher().match_one(read, 0, 44, "V1", indel_count=0)
    assert (m.seq_start, m.seq_end) == (2, 42)            # overhang trimmed back into read coords
    assert (m.ref_start, m.ref_end) == (0, 40)            # full germline span


def test_clean_ends_tolerate_internal_mutations():
    seg = list(_G)
    seg[18] = seg[20] = "N"                                # SHM in the middle, ends still clean
    m = _matcher().match_one("".join(seg), 0, 40, "V1", indel_count=0)
    assert (m.ref_start, m.ref_end) == (0, 40)


def test_from_reference_builds_germline_table():
    class _Ref:
        names = ["A*01", "A*02"]
        sequences = ["acgt", "TTTT"]
    class _RS:
        genes = {"V": _Ref()}
    matcher = HeuristicGermlineMatcher.from_reference(_RS())
    assert matcher._germlines == {"A*01": "ACGT", "A*02": "TTTT"}   # uppercased


# ---------------- derive_alignment (CIGAR reconstruction) ----------------
from alignair.predict.heuristic_matcher import derive_alignment


def _consumes(cigar):
    import re
    q = t = 0
    for num, op in re.findall(r"(\d+)([MID])", cigar):
        num = int(num)
        if op in "MI": q += num
        if op in "MD": t += num
    return q, t


def test_derive_gap_free_is_single_M_run():
    r = derive_alignment("ACGTACGT", "ACGTACGT", indel_count=0)
    assert r.cigar == "8M" and (r.q_end, r.t_end) == (8, 8)


def test_derive_substitution_stays_M():
    r = derive_alignment("ACGT", "AGGT", indel_count=0)      # 1 SHM, no indel
    assert r.cigar == "4M"                                    # M spans matches + mismatches


def test_derive_insertion_places_I():
    r = derive_alignment("ACGGT", "ACGT", indel_count=1)      # read has an extra base
    assert _consumes(r.cigar) == (5, 4)                       # consumes 5 read, 4 germline
    assert "I" in r.cigar and "D" not in r.cigar


def test_derive_deletion_places_D():
    r = derive_alignment("ACT", "ACGT", indel_count=1)        # germline has an extra base
    assert _consumes(r.cigar) == (3, 4)
    assert "D" in r.cigar and "I" not in r.cigar


def test_derive_empty_windows():
    assert derive_alignment("", "ACG", 0).cigar == "3D"
    assert derive_alignment("ACG", "", 0).cigar == "3I"
