import pytest
from alignair.align.parasail import ParasailAligner, parasail_available

pytest.importorskip("pywfa")
from alignair.align.wfa import WFAAligner, wfa_available

pytestmark = pytest.mark.skipif(not (wfa_available() and parasail_available()),
                                reason="pywfa or parasail unavailable")
# (query, target): identity, 5' germline trim, single mismatch, query insertion, germline deletion
CASES = [("ACGTACGT", "ACGTACGT"), ("ACGTACGT", "GGGACGTACGT"),
         ("ACGTACGT", "ACGTTCGT"), ("ACGTAACGT", "ACGTACGT"), ("ACGTCGT", "ACGTACGT")]
# CIGAR is only unambiguous for the substitution-class cases; indel gap PLACEMENT is free
# (WFA and parasail give different-but-equally-optimal cigars), so cigar parity is asserted
# only on the first three. Germline coords (t_start/t_end) are unambiguous on all.
UNAMBIGUOUS_CIGAR = {("ACGTACGT", "ACGTACGT"), ("ACGTACGT", "GGGACGTACGT"), ("ACGTACGT", "ACGTTCGT")}


def test_wfa_germline_coords_match_parasail():
    w, p = WFAAligner(), ParasailAligner()
    for q, t in CASES:
        rw, rp = w.align(q, t), p.align(q, t)
        assert (rw.t_start, rw.t_end) == (rp.t_start, rp.t_end), (q, t)
        assert rw.q_start == 0 and rw.q_end == len(q)
        if (q, t) in UNAMBIGUOUS_CIGAR:
            assert rw.cigar == rp.cigar, (q, t, rw.cigar, rp.cigar)


def test_wfa_cigar_consumes_full_query():
    # the core cigar's M+I (query-consuming ops) must equal the query length
    w = WFAAligner()
    for q, t in CASES:
        r = w.align(q, t)
        import re
        q_consumed = sum(int(n) for n, op in re.findall(r"(\d+)([MI])", r.cigar))
        assert q_consumed == len(q), (q, t, r.cigar)


def test_wfa_ranking_matches_parasail_on_siblings():
    # the operative use: pick the best of several germlines. WFA and parasail must agree on argmax.
    w, p = WFAAligner(), ParasailAligner()
    seg = "ACGTACGTACGTACGT"
    germs = ["ACGTACGTACGTACGT", "ACGTACGTACGTACGA", "TTTTGGGGCCCCAAAA"]
    aw = [w.align(seg, g).score for g in germs]
    ap = [p.align(seg, g).score for g in germs]
    assert aw.index(max(aw)) == ap.index(max(ap)) == 0


def test_too_short_or_empty_returns_none():
    w = WFAAligner()
    assert w.align("", "ACGTACGT") is None
    assert w.align("ACGT", "") is None
