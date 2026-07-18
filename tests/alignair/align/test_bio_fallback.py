"""The Biopython fallback aligner and the get_aligner() backend-selection chain.

Covers the two portability defects: (1) get_aligner must probe each optional backend and never
return one that only fails at align() time, and (2) with NO optional backend (no pywfa, no parasail
- e.g. a default Apple Silicon install) it must still return a working aligner via Biopython, which
is a core dependency."""
from alignair.align import backend
from alignair.align.bio import BioAligner, bio_available


def test_bio_backend_is_always_available():
    assert bio_available() is True


def test_bio_aligner_contract():
    # query embedded in a germline with 5'/3' flanks: query-global, germline ends free
    res = BioAligner().align("ACGTACGTACGT", "TTTTTACGTACGTACGTGGGGG")
    assert res is not None
    assert res.q_start == 0 and res.q_end == 12           # query consumed end to end
    assert res.t_start == 5 and res.t_end == 17           # germline offset where the query sits
    assert set(res.cigar) <= set("0123456789MID")         # core ops only
    assert res.cigar == "12M"                             # exact match, no indels


def test_bio_aligner_rejects_empty():
    assert BioAligner().align("", "ACGT") is None
    assert BioAligner().align("ACGT", "") is None


def test_get_aligner_falls_back_to_biopython_with_no_optional_backend(monkeypatch):
    # simulate a clean environment: neither pywfa nor parasail installed
    monkeypatch.setattr("alignair.align.wfa.wfa_available", lambda: False)
    monkeypatch.setattr("alignair.align.parasail.parasail_available", lambda: False)
    al = backend.get_aligner("wfa")
    assert isinstance(al, BioAligner)
    # and it actually aligns
    res = al.align("ACGTACGTACGT", "TTTACGTACGTACGTTTT")
    assert res is not None and res.q_start == 0 and res.q_end == 12
