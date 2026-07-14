"""Tests for the AIRR assembly subpackage (Phase B)."""
from types import SimpleNamespace

import pytest

from alignair.predict.airr import build_airr
from alignair.predict.airr.builder import AirrAssemblyError
from alignair.predict.airr.alignment import build_sequence_alignment
from alignair.predict.airr.regions import compute_junction


class _FakeRef:
    """Minimal reference stub for status/error-handling tests (no germline math needed)."""
    genes = {"V", "J"}

    def gene(self, g):
        return SimpleNamespace(names=["IGHV1-1*01"], sequences=["ACGT"], gapped=None, anchors={})


def test_build_airr_tags_complete_status(monkeypatch):
    from alignair.predict.airr import builder
    monkeypatch.setattr(builder, "_build_one", lambda rec, *a, **k: dict(rec))
    out = build_airr([{"sequence": "ACGT"}], _FakeRef(), chain="heavy")
    assert out[0]["airr_assembly_status"] == "complete"


def test_build_airr_incomplete_record_is_partial_not_complete():
    """A record that skips assembly (no v_call) is `partial` with a reason code — never a clean
    `complete`/`ok` while germline_alignment/identity are blank (AIRR-review #5)."""
    out = build_airr([{"sequence": "ACGT", "productive": True}], _FakeRef(), chain="heavy")[0]
    assert out["airr_assembly_status"] == "partial"
    assert out["airr_assembly_reason"] == "missing_calls_or_coordinates"


def test_build_airr_unexpected_exception_raises_with_context(monkeypatch):
    """A non-data (programming) exception must fail loudly with the record identifier, never silent."""
    from alignair.predict.airr import builder

    def boom(rec, *a, **k):
        raise TypeError("bug")
    monkeypatch.setattr(builder, "_build_one", boom)
    with pytest.raises(AirrAssemblyError, match="ACGT|record"):
        build_airr([{"sequence": "ACGT"}], _FakeRef(), chain="heavy")


def test_build_airr_expected_exception_is_tagged_not_swallowed(monkeypatch):
    from alignair.predict.airr import builder

    def bad_data(rec, *a, **k):
        raise ValueError("edge-case data")
    monkeypatch.setattr(builder, "_build_one", bad_data)
    out = build_airr([{"sequence": "ACGT", "v_call": "IGHV1-1*01"}], _FakeRef(), chain="heavy")
    assert out[0]["airr_assembly_status"] == "failed"
    assert "ValueError" in out[0]["airr_assembly_error"]
    assert out[0]["v_call"] == "IGHV1-1*01"                 # light record preserved


def test_build_airr_strict_raises_on_expected_error(monkeypatch):
    from alignair.predict.airr import builder
    monkeypatch.setattr(builder, "_build_one", lambda rec, *a, **k: (_ for _ in ()).throw(ValueError("x")))
    with pytest.raises(AirrAssemblyError):
        build_airr([{"sequence": "ACGT"}], _FakeRef(), chain="heavy", strict=True)


def test_build_airr_productive_blank_when_not_derivable():
    """A record that skips assembly (no v_call) leaves AIRR `productive` BLANK (unknown), keeping the
    neural call only in `productive_prediction` — no guess presented as a derived fact (audit #6)."""
    out = build_airr([{"sequence": "ACGT", "productive": True}], _FakeRef(), chain="heavy")[0]
    assert out["productive"] is None                    # underivable -> blank/unknown
    assert out["productive_prediction"] is True         # the neural prediction is preserved separately


def test_sequence_alignment_inserts_imgt_gaps():
    # gapped V ref has gaps at positions 2 and 5; the query (== ungapped V) is re-gapped to match.
    aln = build_sequence_alignment("ACGTACGT", "AC.GT.ACGT", 0, 8, 0, 8, j_seq_end=8)
    assert aln == "AC.GT.ACGT"


def test_junction_end_from_anchor():
    seq_alignment = "A" * 400                      # no gaps -> offset 0
    seq_aa = "K" * 133
    j = compute_junction(seq_alignment, seq_aa, "J1", {"J1": 30},
                         j_seq_start=350, v_seq_start=0, j_germ_start=0, j_alignment_end=400)
    # junc_end = 350 - 0 + 0 + 30 - 0 + 3 = 383 ; junction = [309:383]
    assert j["junction_length"] == 74
    assert j["cdr3_start"] == 312 and j["cdr3_end"] == 380


@pytest.mark.slow
def test_build_airr_end_to_end_produces_regions_and_junction():
    import GenAIRR.data as gd
    from alignair.reference.reference_set import ReferenceSet
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    v0, j0 = ref.gene("V").names[0], ref.gene("J").names[0]
    v_seq = ref.gene("V").sequences[ref.gene("V").index[v0]]
    j_seq = ref.gene("J").sequences[ref.gene("J").index[j0]]
    read = v_seq + "NNNNN" + j_seq
    rec = {"sequence_id": "t0", "sequence": read, "v_call": v0, "d_call": "Short-D", "j_call": j0,
           "v_sequence_start": 0, "v_sequence_end": len(v_seq),
           "v_germline_start": 0, "v_germline_end": len(v_seq),
           "d_sequence_start": len(v_seq), "d_sequence_end": len(v_seq) + 5,
           "d_germline_start": 0, "d_germline_end": 0,
           "j_sequence_start": len(v_seq) + 5, "j_sequence_end": len(read),
           "j_germline_start": 0, "j_germline_end": len(j_seq),
           "productive": True, "indel_count": 0}
    out = build_airr([rec], ref, chain="heavy")[0]
    assert out["sequence_alignment"] is not None and "." in out["sequence_alignment"]  # V is IMGT-gapped
    assert out["fwr1"] is not None and out["cdr1"] is not None
    assert out["junction"] is not None and out["junction_length"] > 0
    assert out["v_germline_start"] == 0
