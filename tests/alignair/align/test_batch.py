import pytest
from alignair.align.batch import align_batch
from alignair.align.parasail import ParasailAligner, parasail_available

pytestmark = pytest.mark.skipif(not parasail_available(), reason="parasail not installed")


def test_batch_preserves_order_and_matches_serial():
    al = ParasailAligner()
    pairs = [("ACGTACGT", "ACGTACGT"), ("ACGTACGT", "ACGTTCGT"), ("ACGTCGT", "ACGTACGT")]
    got = align_batch(pairs, al, workers=2)
    exp = [al.align(q, t) for q, t in pairs]
    assert [r.cigar for r in got] == [r.cigar for r in exp]
    assert [r.score for r in got] == [r.score for r in exp]


def test_batch_handles_none_results():
    al = ParasailAligner()
    got = align_batch([("", "ACGT"), ("ACGTACGT", "ACGTACGT")], al, workers=2)
    assert got[0] is None and got[1] is not None


def test_empty_batch():
    assert align_batch([], ParasailAligner()) == []
