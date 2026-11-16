"""Tests for the AIRR assembly subpackage."""
from types import SimpleNamespace

import pytest

from alignair.predict.airr import build_airr
from alignair.predict.airr.builder import AirrAssemblyError
from alignair.predict.airr.alignment import build_sequence_alignment
from alignair.predict.results import Anchors
from alignair.predict.state import GermlineAlignment
from tests.support.read_results import make_read

pytestmark = pytest.mark.component


class _FakeRef:
    """Minimal reference stub for status/error-handling tests (no germline math needed)."""
    genes = {"V", "J"}
    loci = []                                       # no locus schema

    def gene(self, g):
        return SimpleNamespace(names=["IGHV1-1*01"], sequences=["ACGT"], gapped=None, anchors={})


def test_a_model_without_a_d_never_assembles_one_whatever_the_reference_says(monkeypatch):
    """A model without a D cannot call one: its reads are assembled without a D even when the bound reference gives
    their locus one (a D-less model bound to an IGH reference)."""
    from alignair.predict.airr import builder
    seen = []
    monkeypatch.setattr(builder, "_build_one", lambda read, light, germlines, has_d: seen.append(has_d) or dict(light))
    reference = _FakeRef()
    reference.loci = [SimpleNamespace(locus="IGH", has_d=True)]
    build_airr([make_read("ACGT", locus="IGH"), make_read("ACGT")], reference, has_d=False)
    assert seen == [False, False]
    seen.clear()
    build_airr([make_read("ACGT", locus="IGH"), make_read("ACGT")], reference, has_d=True)
    assert seen == [True, True]


def test_build_airr_tags_complete_status(monkeypatch):
    from alignair.predict.airr import builder
    monkeypatch.setattr(builder, "_build_one", lambda read, light, *a, **k: dict(light))
    out = build_airr([make_read("ACGT")], _FakeRef(), has_d=True)
    assert out[0]["airr_assembly_status"] == "complete"


def test_build_airr_incomplete_record_is_partial_not_complete():
    """A record that skips assembly (no v_call) is `partial` with a reason code — never a clean
    `complete`/`ok` while germline_alignment/identity are blank."""
    out = build_airr([make_read("ACGT")], _FakeRef(), has_d=True)[0]
    assert out["airr_assembly_status"] == "partial"
    assert out["airr_assembly_reason"] == "missing_calls_or_coordinates"


def test_build_airr_unexpected_exception_raises_with_context(monkeypatch):
    """A non-data (programming) exception must fail loudly with the record identifier, never silent."""
    from alignair.predict.airr import builder

    def boom(read, light, *a, **k):
        raise TypeError("bug")
    monkeypatch.setattr(builder, "_build_one", boom)
    with pytest.raises(AirrAssemblyError, match="ACGT|record"):
        build_airr([make_read("ACGT")], _FakeRef(), has_d=True)


def test_build_airr_expected_exception_is_tagged_not_swallowed(monkeypatch):
    from alignair.predict.airr import builder

    def bad_data(read, light, *a, **k):
        raise ValueError("edge-case data")
    monkeypatch.setattr(builder, "_build_one", bad_data)
    out = build_airr([make_read("ACGT", calls={"v": "IGHV1-1*01"})], _FakeRef(), has_d=True)
    assert out[0]["airr_assembly_status"] == "failed"
    assert "ValueError" in out[0]["airr_assembly_error"]
    assert out[0]["v_call"] == "IGHV1-1*01"                 # light record preserved


def test_build_airr_strict_raises_on_expected_error(monkeypatch):
    from alignair.predict.airr import builder
    monkeypatch.setattr(builder, "_build_one", lambda read, light, *a, **k: (_ for _ in ()).throw(ValueError("x")))
    with pytest.raises(AirrAssemblyError):
        build_airr([make_read("ACGT")], _FakeRef(), has_d=True, strict=True)


def test_build_airr_productive_blank_without_anchors():
    """A record without predicted anchors has no junction, frame or productivity: AIRR `productive` is blank
    (unknown), never a guess presented as a derived fact."""
    out = build_airr([make_read("ACGT")], _FakeRef(), has_d=True)[0]
    assert out["productive"] is None and out["productive_prediction"] is None
    assert "junction" not in out


def test_sequence_alignment_inserts_imgt_gaps():
    # gapped V ref has gaps at positions 2 and 5; the query (== ungapped V) is re-gapped to match.
    aln = build_sequence_alignment("ACGTACGT", "AC.GT.ACGT", GermlineAlignment("V1", 0, 8, 0, 8, ""), 8)
    assert aln == "AC.GT.ACGT"


@pytest.mark.slow
def test_build_airr_end_to_end_produces_regions_and_junction():
    import GenAIRR.data as gd
    from alignair.reference.reference_set import ReferenceSet
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    v0, j0 = ref.gene("V").names[0], ref.gene("J").names[0]
    v_seq = ref.gene("V").sequences[ref.gene("V").index[v0]]
    j_seq = ref.gene("J").sequences[ref.gene("J").index[j0]]
    read = v_seq + "NNNNN" + j_seq
    n = len(v_seq)
    rec = make_read(read, calls={"v": v0, "j": j0}, genes=("v", "d", "j"), d_presence="short", alignments={
        "v": GermlineAlignment(v0, 0, n, 0, n, ""), "d": GermlineAlignment("", n, n + 5, 0, 0, ""),
        "j": GermlineAlignment(j0, n + 5, len(read), 0, len(j_seq), "")},
        anchors=Anchors(ref.gene("V").anchors[v0], n + 5 + ref.gene("J").anchors[j0]))
    out = build_airr([rec], ref, has_d=True)[0]
    assert out["sequence_alignment"] is not None and "." in out["sequence_alignment"]  # V is IMGT-gapped
    assert out["fwr1"] is not None and out["cdr1"] is not None
    assert out["junction"] is not None and out["junction_length"] > 0
    assert out["v_germline_start"] == 0


@pytest.mark.slow
def test_build_airr_keeps_the_aligners_v_germline_coordinates():
    """``v_germline_start``/``end`` are positions in the germline, as the aligner gives them: the IMGT gaps of the
    alignment strings do not move them, and a V cut at its 5' end keeps where its germline starts."""
    import GenAIRR.data as gd
    from alignair.reference.reference_set import ReferenceSet
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    v0 = next(name for name in ref.gene("V").names if "." in ref.gene("V").gapped[name])
    j0 = ref.gene("J").names[0]
    v_seq = ref.gene("V").sequences[ref.gene("V").index[v0]]
    j_seq = ref.gene("J").sequences[ref.gene("J").index[j0]]
    cut = 30
    read = v_seq[cut:] + "NNNNN" + j_seq
    v_len = len(v_seq) - cut
    rec = make_read(read, calls={"v": v0, "j": j0}, genes=("v", "d", "j"), d_presence="short", alignments={
        "v": GermlineAlignment(v0, 0, v_len, cut, len(v_seq), ""),
        "d": GermlineAlignment("", v_len, v_len + 5, 0, 0, ""),
        "j": GermlineAlignment(j0, v_len + 5, len(read), 0, len(j_seq), "")})
    out = build_airr([rec], ref, has_d=True)[0]
    assert (out["v_germline_start"], out["v_germline_end"]) == (cut, len(v_seq))


@pytest.mark.slow
def test_a_row_is_its_light_record_plus_the_assembly():
    """``build_airr`` assembles a read's typed result: the row starts from its light record (every light field as
    ``light_record`` gives it, ``productive`` aside, which the assembly may fill) and adds the assembly."""
    import GenAIRR.data as gd
    from alignair.predict.results import light_record
    from alignair.predict.state import GermlineAlignment
    from alignair.reference.reference_set import ReferenceSet
    from tests.support.read_results import make_read
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    v0, j0 = ref.gene("V").names[0], ref.gene("J").names[0]
    v_seq = ref.gene("V").sequences[ref.gene("V").index[v0]]
    j_seq = ref.gene("J").sequences[ref.gene("J").index[j0]]
    read_seq = v_seq + "NNNNN" + j_seq
    n = len(v_seq)
    read = make_read(read_seq, calls={"v": v0, "j": j0}, genes=("v", "d", "j"), d_presence="short", alignments={
        "v": GermlineAlignment(v0, 0, n, 0, n, f"{n}M"), "d": GermlineAlignment("", n, n + 5, 0, 0, ""),
        "j": GermlineAlignment(j0, n + 5, len(read_seq), 0, len(j_seq), f"{len(j_seq)}M")})
    (row,) = build_airr([read], ref, has_d=True)
    assert all(row[key] == value for key, value in light_record(read).items() if key != "productive")
    assert row["airr_assembly_status"] == "complete" and row["sequence_alignment"]
