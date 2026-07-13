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

def test_writer_emits_canonical_sequence_not_external_input():
    """A reoriented read: the record owns the canonical sequence the coords are in; the writer must
    emit THAT (not the separately-passed original input), and preserve the input as provenance."""
    canonical = "ACGTACGTAC"
    original = "GTACGTACGT"                    # pre-orientation read, a different string
    rec = {"sequence": canonical, "orientation_id": 1, "v_call": "IGHV1-1*01",
           "v_sequence_start": 0, "v_sequence_end": 10}
    tmp = tempfile.mktemp(suffix=".tsv")
    try:
        write_airr(tmp, ["r"], [original], [rec])     # external list carries the ORIGINAL input
        row = _read_rows(tmp)[0]
        assert row["sequence"] == canonical            # coordinates refer to this exact string
        assert row["input_sequence"] == original       # original read preserved for provenance
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
