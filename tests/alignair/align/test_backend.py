import pytest
from alignair.align.backend import AlignResult, get_aligner


def test_align_result_fields():
    r = AlignResult(score=16.0, cigar="8M", q_start=0, q_end=8, t_start=0, t_end=8)
    assert r.score == 16.0 and r.cigar == "8M"
    assert r.q_start == 0 and r.q_end == 8 and r.t_start == 0 and r.t_end == 8


def test_get_aligner_returns_something_with_align():
    al = get_aligner()                       # falls back to parasail if wfa absent
    assert hasattr(al, "align")
    r = al.align("ACGTACGT", "ACGTACGT")
    assert r is not None and r.cigar == "8M"
