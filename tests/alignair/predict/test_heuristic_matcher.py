"""Behaviour tests for the heuristic germline matcher (clean port of TF HeuristicReferenceMatcher).

The exhaustive identical-results check vs the TF logic was run over the whole benchmark
(7920 matches, V/D/J x fractional indel counts) at port time; these lock in the key behaviours.
"""
import pytest
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


_LONG = "".join(__import__("random").Random(7).choices("ACGT", k=300))   # a 300 bp germline


def test_a_body_deletion_under_an_underpredicted_indel_count_keeps_the_anchored_start():
    # full-length V with one base deleted at 100: the read's 3' majority sits one germline base later, but its
    # start is still germline base 0; the model's indel count (0.1) rounds to none
    read = _LONG[:100] + _LONG[101:]
    m = HeuristicGermlineMatcher({"V1": _LONG}).match_one(read, 0, len(read), "V1", indel_count=0.1)
    assert (m.ref_start, m.ref_end) == (0, 300)


def test_three_body_deletions_under_an_underpredicted_indel_count_keep_the_start():
    # three bases deleted (at 80, 160 and 240): the length gives a start 3 bases in, and the model's indel count
    # (0.1) rounds to none, yet the read's first bases are the germline's first
    read = _LONG[:80] + _LONG[81:160] + _LONG[161:240] + _LONG[241:]
    m = HeuristicGermlineMatcher({"V1": _LONG}).match_one(read, 0, len(read), "V1", indel_count=0.1)
    assert (m.ref_start, m.ref_end) == (0, 300)


def test_a_body_insertion_keeps_the_germline_end_of_a_3prime_trimmed_v():
    # one base inserted at 100 and the V trimmed 3 bases short of its germline end: the read is only 2 bases shorter
    # than the germline, yet its last base is germline base 296
    read = _LONG[:100] + "A" + _LONG[100:297]
    m = HeuristicGermlineMatcher({"V1": _LONG}).match_one(read, 0, len(read), "V1", indel_count=0.1)
    assert (m.ref_start, m.ref_end) == (0, 297)


def test_a_segment_longer_than_its_germline_by_insertions_keeps_its_read_end():
    # two bases inserted at 100 and the V trimmed one base short: 301 read bases over germline bases 0..298; the
    # excess over the germline's length is insertions, not overhang, so the read's V still ends at its last base
    read = _LONG[:100] + "AC" + _LONG[100:299]
    m = HeuristicGermlineMatcher({"V1": _LONG}).match_one(read, 0, len(read), "V1", indel_count=0.1)
    assert (m.seq_start, m.seq_end, m.ref_start, m.ref_end) == (0, 301, 0, 299)


def test_a_segment_running_past_its_germline_into_other_bases_spans_the_whole_germline():
    # a segment that runs 12 bases past the germline's 3' end (into a constant region, say): its tail is not germline
    # wherever the end anchor lands, so the excess is overhang and the germline span is the whole germline
    read = _LONG + "".join(__import__("random").Random(11).choices("ACGT", k=12))
    m = HeuristicGermlineMatcher({"V1": _LONG}).match_one(read, 0, len(read), "V1", indel_count=0.1)
    assert (m.ref_start, m.ref_end) == (0, 300)


def test_a_3prime_truncated_amplicon_relocates_its_start_by_the_scan():
    # an FR1-like read ending 180 bases before the germline's 3' end, beyond the end anchor's reach
    read = _LONG[:120]
    m = HeuristicGermlineMatcher({"V1": _LONG}).match_one(read, 0, 120, "V1", indel_count=0)
    assert m.ref_start == 0


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

pytestmark = pytest.mark.component


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


def test_derive_places_a_balanced_insertion_and_deletion_the_indel_count_missed():
    # a base inserted at 50 and one deleted at 150 keep the window lengths equal; the model's count rounds to none
    germ = _LONG[:200]
    read = germ[:50] + "A" + germ[50:150] + germ[151:]
    r = derive_alignment(read, germ, indel_count=0.03)
    assert "I" in r.cigar and "D" in r.cigar and _consumes(r.cigar) == (200, 200)


def test_derive_keeps_spread_substitutions_as_matches_when_the_lengths_agree():
    # ten SHM substitutions 20 bases apart: shifting any stretch between them costs more than it saves
    read = list(_LONG[:200])
    for i in range(5, 200, 20):
        read[i] = "A" if read[i] != "A" else "C"
    assert derive_alignment("".join(read), _LONG[:200], indel_count=0).cigar == "200M"


def test_derive_empty_windows():
    assert derive_alignment("", "ACG", 0).cigar == "3D"
    assert derive_alignment("ACG", "", 0).cigar == "3I"


def test_the_compiled_kernel_matches_the_python_reference():
    cy = pytest.importorskip("alignair.predict._derive_cy")
    from alignair.predict.heuristic_matcher import _derive_cigar_py
    rng = __import__("random").Random(3)
    for _ in range(400):
        germ = "".join(rng.choices("ACGT", k=rng.randint(0, 80)))
        read = list(germ)
        for _ in range(rng.randint(0, 8)):                  # substitutions, insertions and deletions
            at, kind = rng.randint(0, len(read)), rng.choice("SID")
            if kind == "I":
                read.insert(at, rng.choice("ACGT"))
            elif read and at < len(read):
                if kind == "S":
                    read[at] = rng.choice("ACGT")
                else:
                    del read[at]
        read, indels = "".join(read), rng.randint(0, 3)
        assert cy.derive_cigar(read.encode(), germ.encode(), indels) == _derive_cigar_py(read, germ, indels)
