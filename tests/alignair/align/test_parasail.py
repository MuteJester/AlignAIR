import pytest
from alignair.align.parasail import ParasailAligner, parasail_available

pytestmark = pytest.mark.skipif(not parasail_available(), reason="parasail not installed")
AL = ParasailAligner()


def test_identity():
    r = AL.align("ACGTACGT", "ACGTACGT")
    assert (r.cigar, r.t_start, r.t_end) == ("8M", 0, 8)
    assert r.q_start == 0 and r.q_end == 8 and r.score == 16


def test_germline_5p_trim_sets_t_start():
    r = AL.align("ACGTACGT", "GGGACGTACGT")     # 3 free germline bases on the 5' end
    assert (r.cigar, r.t_start, r.t_end) == ("8M", 3, 11)
    assert r.score == 16


def test_single_mismatch_is_M_with_lower_score():
    r = AL.align("ACGTACGT", "ACGTTCGT")
    assert (r.cigar, r.t_start, r.t_end) == ("8M", 0, 8)
    assert r.score == 13


def test_query_insertion():
    r = AL.align("ACGTAACGT", "ACGTACGT")        # extra query A -> insertion (query-only)
    assert (r.cigar, r.t_start, r.t_end) == ("4M1I4M", 0, 8)
    assert r.q_end == 9 and r.score == 13


def test_germline_deletion():
    r = AL.align("ACGTCGT", "ACGTACGT")          # missing query A -> deletion (germline-only)
    assert (r.cigar, r.t_start, r.t_end) == ("4M1D3M", 0, 8)
    assert r.q_end == 7 and r.score == 11


def test_too_short_or_empty_returns_none():
    assert AL.align("", "ACGTACGT") is None
    assert AL.align("ACGT", "") is None
