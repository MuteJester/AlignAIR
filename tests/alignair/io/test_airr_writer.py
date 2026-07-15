"""AIRR rearrangement writer: full-schema columns, AIRR conventions (1-based coords, rev_comp flag),
and pass-through of the assembled AIRR biology (junction / regions)."""
import csv
import os
import tempfile

import pytest

from alignair.io.airr import AirrWriter, COLUMN_PRESETS, COLUMNS, needs_assembly, resolve_columns, write_airr


def _read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_writer_full_airr_roundtrip():
    rec = {
        "sequence": "ACGTACGTACGT", "orientation_id": 1, "productive": True,
        "v_call": "IGHV1-1*01", "d_call": "IGHD1-1*01", "j_call": "IGHJ1*01",
        "v_calls": ["IGHV1-1*01", "IGHV1-1*02"],
        "v_sequence_start": 0, "v_sequence_end": 8, "v_germline_start": 2, "v_germline_end": 10,
        "v_cigar": "8M", "junction": "TGTGCG", "junction_aa": "CA", "junction_length": 6,
        "fwr1": "ACGT", "cdr3": "GCG", "v_identity": 0.99, "sequence_alignment": "ACGTACGT",
    }
    tmp = tempfile.mktemp(suffix=".tsv")
    try:
        write_airr(tmp, ["read1"], [rec["sequence"]], [rec])
        rows = _read_rows(tmp)
        assert len(rows) == 1
        row = rows[0]
        assert row["sequence_id"] == "read1"
        assert row["rev_comp"] == "T"                 # orientation_id 1 -> reoriented
        assert row["v_sequence_start"] == "1"         # 0-based 0 -> 1-based 1
        assert row["v_germline_start"] == "3"         # 0-based 2 -> 1-based 3
        assert row["v_sequence_end"] == "8"           # AIRR end (inclusive) == our end-exclusive
        assert row["v_cigar"] == "8M"
        assert row["junction"] == "TGTGCG"            # assembled AIRR biology passes through
        assert row["cdr3"] == "GCG" and row["fwr1"] == "ACGT"
        assert row["v_call_set"] == "IGHV1-1*01,IGHV1-1*02"
    finally:
        os.remove(tmp)


def test_writer_forward_read_is_not_rev_comp():
    rec = {"sequence": "ACGT", "orientation_id": 0, "v_call": "IGHV1-1*01",
           "v_sequence_start": 0, "v_sequence_end": 4}
    tmp = tempfile.mktemp(suffix=".tsv")
    try:
        write_airr(tmp, ["r"], ["ACGT"], [rec])
        assert _read_rows(tmp)[0]["rev_comp"] == "F"
    finally:
        os.remove(tmp)


# --- P0-3: AIRR coordinates always refer to the emitted (canonical) sequence --------------------

_COMPL = str.maketrans("ACGTN", "TGCAN")


def _rc(s):
    return s.translate(_COMPL)[::-1]


def test_rev_comp_output_follows_airr_convention():
    """AIRR: for a reverse-complement hit, `sequence` is the ORIGINAL query and all alignment data are
    based on RC(sequence) == the aligned (canonical) frame. The writer must emit the original, so a
    consumer that reverse-complements `sequence` recovers exactly the frame the coordinates are in."""
    aligned = "ACGTACGTAC"                     # the frame coordinates/alignments are computed in
    original = _rc(aligned)                     # the read as submitted (reverse-complemented)
    rec = {"sequence": aligned, "orientation_id": 1, "v_call": "IGHV1-1*01",
           "v_sequence_start": 0, "v_sequence_end": 10}
    tmp = tempfile.mktemp(suffix=".tsv")
    try:
        write_airr(tmp, ["r"], [original], [rec])     # external list carries the ORIGINAL input
        row = _read_rows(tmp)[0]
        assert row["rev_comp"] == "T"
        assert row["sequence"] == original             # the query as submitted
        assert _rc(row["sequence"]) == aligned         # RC(sequence) == the aligned/coordinate frame
    finally:
        os.remove(tmp)


def test_rev_comp_uses_record_owned_input_sequence_not_external():
    """The emitted original query is the RECORD's own `input_sequence`, so a stale/wrong external
    sequence cannot corrupt orientation output — this is the Python-API double-reverse defect fixed."""
    aligned = "ACGTACGTAC"                          # coordinate frame (canonical)
    original = _rc(aligned)                          # post-crop, pre-orientation read the record owns
    rec = {"sequence": aligned, "input_sequence": original, "orientation_id": 1,
           "v_call": "IGHV1-1*01", "v_sequence_start": 0, "v_sequence_end": 10}
    tmp = tempfile.mktemp(suffix=".tsv")
    try:
        write_airr(tmp, ["r"], ["WRONG-EXTERNAL-SEQUENCE"], [rec])   # bogus external list
        row = _read_rows(tmp)[0]
        assert row["rev_comp"] == "T"
        assert row["sequence"] == original           # record-owned, NOT the bogus external
        assert _rc(row["sequence"]) == aligned        # RC(sequence) == the coordinate frame
    finally:
        os.remove(tmp)


def test_cropped_rev_comp_sequence_matches_coordinate_frame():
    """For a cropped reverse-complement read the emitted `sequence` is the POST-crop original, so
    RC(sequence) equals the cropped coordinate frame — the uncropped external read must not leak in."""
    aligned = "ACGTACGTAC"                          # cropped canonical (coordinate frame)
    cropped_original = _rc(aligned)                  # post-crop, pre-orientation
    uncropped = cropped_original + "TTTTGGGGAAAA"    # the full read the CLI passes externally
    rec = {"sequence": aligned, "input_sequence": cropped_original, "orientation_id": 1,
           "length_cropped": True, "v_call": "IGHV1-1*01", "v_sequence_start": 0, "v_sequence_end": 10}
    tmp = tempfile.mktemp(suffix=".tsv")
    try:
        write_airr(tmp, ["r"], [uncropped], [rec])   # CLI hands over the uncropped read
        row = _read_rows(tmp)[0]
        assert row["sequence"] == cropped_original
        assert _rc(row["sequence"]) == aligned        # RC == cropped coordinate frame (no leak)
        assert row["length_cropped"] == "T"
    finally:
        os.remove(tmp)


def test_cli_and_prediction_result_orientation_fields_agree():
    """The CLI (AirrWriter + external seq list) and the Python API (PredictionResult.write_airr) must
    emit identical orientation fields, because both defer to the record-owned input_sequence."""
    from alignair import PredictionResult
    aligned = "ACGTACGTAC"
    original = _rc(aligned)
    rec = {"sequence": aligned, "input_sequence": original, "orientation_id": 1, "locus": "IGH",
           "v_call": "IGHV1-1*01", "v_sequence_start": 0, "v_sequence_end": 10}
    p_tmp, c_tmp = tempfile.mktemp(suffix=".tsv"), tempfile.mktemp(suffix=".tsv")
    try:
        PredictionResult([dict(rec)], locus="IGH").write_airr(p_tmp)             # Python API
        write_airr(c_tmp, ["seq0"], ["A-DIFFERENT-EXTERNAL"], [dict(rec)], locus="IGH")  # CLI-style
        pr, cr = _read_rows(p_tmp)[0], _read_rows(c_tmp)[0]
        for f in ("sequence", "rev_comp", "orientation", "input_sequence"):
            assert pr.get(f) == cr.get(f), f
        assert pr["sequence"] == original and pr["rev_comp"] == "T"
    finally:
        os.remove(p_tmp)
        os.remove(c_tmp)


def test_complement_only_read_emits_canonical_with_rev_comp_false():
    """rev_comp can only encode reverse-complement; a complement-only read emits the canonical frame
    (coords valid on it) with rev_comp=F and the true transform in `orientation`."""
    aligned = "ACGTACGTAC"
    original = aligned.translate(_COMPL)              # complement-only (id 2), not a reverse-complement
    rec = {"sequence": aligned, "orientation_id": 2, "v_call": "IGHV1-1*01",
           "v_sequence_start": 0, "v_sequence_end": 10}
    tmp = tempfile.mktemp(suffix=".tsv")
    try:
        write_airr(tmp, ["r"], [original], [rec])
        row = _read_rows(tmp)[0]
        assert row["rev_comp"] == "F" and row["orientation"] == "complement"
        assert row["sequence"] == aligned and row["input_sequence"] == original
    finally:
        os.remove(tmp)


def test_rev_comp_true_only_for_reverse_complement_orientation():
    """AIRR ``rev_comp`` means reverse-complement (orientation id 1). Complement-only (2) and
    reverse-only (3) are NOT rev-comp; the full transform is preserved in the ``orientation`` field."""
    expect = {0: ("F", "forward"), 1: ("T", "reverse_complement"),
              2: ("F", "complement"), 3: ("F", "reverse")}
    for oid, (rc, label) in expect.items():
        rec = {"sequence": "ACGT", "orientation_id": oid, "v_call": "IGHV1-1*01",
               "v_sequence_start": 0, "v_sequence_end": 4}
        tmp = tempfile.mktemp(suffix=".tsv")
        try:
            write_airr(tmp, ["r"], ["ACGT"], [rec])
            row = _read_rows(tmp)[0]
            assert row["rev_comp"] == rc, oid
            assert row["orientation"] == label, oid
        finally:
            os.remove(tmp)


def test_writer_header_is_full_airr_schema():
    tmp = tempfile.mktemp(suffix=".tsv")
    try:
        write_airr(tmp, [], [], [])
        header = open(tmp).readline().strip().split("\t")
        assert header == COLUMNS
        for col in ("junction", "junction_aa", "fwr1", "cdr3", "sequence_alignment",
                    "v_cigar", "v_sequence_start", "v_identity", "productive"):
            assert col in header
    finally:
        os.remove(tmp)


def test_column_presets_and_custom_resolve():
    assert resolve_columns(None) == COLUMNS                      # default = full
    assert resolve_columns("full") == COLUMNS
    assert resolve_columns("minimal") == COLUMN_PRESETS["minimal"]
    assert resolve_columns("v_call,junction, j_call") == ["v_call", "junction", "j_call"]  # comma-string
    assert resolve_columns(["v_call", "d_call"]) == ["v_call", "d_call"]                    # explicit list


def test_needs_assembly_skips_for_light_selections():
    assert needs_assembly(None) is True                          # full needs build_airr
    assert needs_assembly("core") is True                        # includes junction
    assert needs_assembly("minimal") is False                    # calls + productive only
    assert needs_assembly("v_call,j_call") is False
    assert needs_assembly("v_call,cdr3") is True                 # a region field needs assembly


def test_writer_rejects_prediction_count_mismatch():
    """Fewer/extra predictions than inputs must RAISE (never zip-truncate + silently drop reads);
    the atomic output is discarded (AIRR-review #1)."""
    from alignair.io.airr import AirrWriter, PredictionCountMismatch
    tmp = tempfile.mktemp(suffix=".tsv")
    try:
        with pytest.raises(PredictionCountMismatch):
            with AirrWriter(tmp) as w:
                w.write(["r1", "r2", "r3"], ["A", "C", "G"], [{"sequence": "A"}, {"sequence": "C"}])
        assert not os.path.exists(tmp)                 # discarded, not a truncated partial file
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def test_writer_carries_metadata_columns():
    """Per-read metadata (barcode/UMI/cell_id) is carried into extra output columns for 10x/Scirpy."""
    rec = {"sequence": "ACGT", "v_call": "IGHV1-1*01"}
    tmp = tempfile.mktemp(suffix=".tsv")
    try:
        write_airr(tmp, ["c1"], ["ACGT"], [rec], columns="minimal",
                   metas=[{"barcode": "AAAA-1", "umi_count": "7"}], extra_columns=["barcode", "umi_count"])
        row = _read_rows(tmp)[0]
        assert row["barcode"] == "AAAA-1" and row["umi_count"] == "7" and row["v_call"] == "IGHV1-1*01"
    finally:
        os.remove(tmp)


def test_metadata_cannot_overwrite_model_fields():
    """Adversarial: metadata columns colliding with produced fields (v_call/sequence/productive) must
    NOT clobber the aligner's result (AIRR-review)."""
    rec = {"sequence": "ACGT", "v_call": "IGHV1-1*01", "productive": "T"}
    tmp = tempfile.mktemp(suffix=".tsv")
    try:
        write_airr(tmp, ["r"], ["ACGT"], [rec], columns="minimal",
                   metas=[{"v_call": "EVIL*99", "sequence": "GGGG", "cell_id": "AAAA-1"}],
                   extra_columns=["cell_id"])
        row = _read_rows(tmp)[0]
        assert row["v_call"] == "IGHV1-1*01" and row["sequence"] == "ACGT"   # model result preserved
        assert row["cell_id"] == "AAAA-1"                                    # new column added
    finally:
        os.remove(tmp)


def test_metadata_fills_blank_c_call():
    """c_call is fill-only: when the record has no constant-region call, the 10x c_gene (normalized to
    c_call) fills it (AIRR-review correction 1)."""
    rec = {"sequence": "ACGT", "v_call": "IGHV1-1*01"}                # no c_call produced by the model
    tmp = tempfile.mktemp(suffix=".tsv")
    try:
        write_airr(tmp, ["c1"], ["ACGT"], [rec], columns="full", metas=[{"c_call": "IGHM"}])
        assert _read_rows(tmp)[0]["c_call"] == "IGHM"
    finally:
        os.remove(tmp)


def test_metadata_does_not_overwrite_populated_c_call():
    """c_call is fill-ONLY: a value already on the record is preserved, not clobbered by metadata."""
    rec = {"sequence": "ACGT", "v_call": "IGHV1-1*01", "c_call": "IGHG"}   # record already has c_call
    tmp = tempfile.mktemp(suffix=".tsv")
    try:
        write_airr(tmp, ["c1"], ["ACGT"], [rec], columns="full", metas=[{"c_call": "IGHM"}])
        assert _read_rows(tmp)[0]["c_call"] == "IGHG"                  # not overwritten
    finally:
        os.remove(tmp)


def test_metadata_dangerous_call_column_is_namespaced():
    """A metadata column colliding with a produced call (v_call) is namespaced to meta_v_call by the
    CLI's rename policy, so it can never masquerade as the model's call. c_call is the sole exception."""
    from alignair.io.airr import _METADATA_FILL_ONLY, COLUMNS
    protected = frozenset(COLUMNS)

    def _safe(col):                                 # mirrors cli.predict._load_metadata_table
        if col in _METADATA_FILL_ONLY:
            return col
        return f"meta_{col}" if col in protected else col

    assert _safe("v_call") == "meta_v_call"         # dangerous call -> namespaced
    assert _safe("sequence") == "meta_sequence"
    assert _safe("c_call") == "c_call"              # fill-only exception keeps its AIRR name
    assert _safe("cell_id") == "cell_id"            # non-protected passes through


def test_writer_is_atomic_on_interrupt():
    """An interrupted write must not leave the final path (only a discarded temp), so a crashed job is
    never mistaken for a complete one (P0-8)."""
    rec = {"sequence": "ACGT", "v_call": "IGHV1-1*01"}
    tmp = tempfile.mktemp(suffix=".tsv")
    with pytest.raises(RuntimeError):
        with AirrWriter(tmp) as w:
            w.write(["r"], ["ACGT"], [rec])
            raise RuntimeError("boom")                 # simulate an interrupted job
    assert not os.path.exists(tmp)                     # final path never appeared
    assert not os.path.exists(f"{tmp}.tmp.{os.getpid()}")  # temp cleaned up


def test_writer_emits_only_selected_columns():
    rec = {"sequence": "ACGT", "v_call": "IGHV1-1*01", "d_call": "IGHD1-1*01", "j_call": "IGHJ1*01",
           "productive": True, "junction": "TGT"}
    tmp = tempfile.mktemp(suffix=".tsv")
    try:
        write_airr(tmp, ["r"], ["ACGT"], [rec], columns="minimal")
        header = open(tmp).readline().strip().split("\t")
        assert header == COLUMN_PRESETS["minimal"]
        assert "junction" not in header and "v_cigar" not in header
    finally:
        os.remove(tmp)
