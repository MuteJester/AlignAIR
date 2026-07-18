"""Interop smoke test: AlignAIR's AIRR output validates against the official ``airr`` schema and carries
the columns the common downstream importers read (Scirpy, Change-O / Immcantation, nf-core/airrflow).

The external tools themselves are not installed/run here; this asserts the AlignAIR side of the
integration contract - schema validity plus required-column coverage - which is what those tools import
against. A value may still be blank when it is genuinely underivable (e.g. ``junction`` on a short
fragment) or metadata-supplied (e.g. ``cell_id``); this test checks the columns are emitted, and that a
complete record validates.
"""
import os
import tempfile

import pytest

from alignair.io.airr import COLUMNS, write_airr

airr = pytest.importorskip("airr")


# Columns each downstream importer reads from an AIRR rearrangement TSV. AlignAIR must EMIT these.
_MIAIRR_REQUIRED = {  # nf-core/airrflow assembled-mode / MiAIRR rearrangement minimum
    "sequence_id", "sequence", "rev_comp", "productive",
    "v_call", "d_call", "j_call", "sequence_alignment", "germline_alignment",
    "junction", "junction_aa", "v_cigar", "d_cigar", "j_cigar",
}
_SCIRPY_FIELDS = {"sequence_id", "locus", "v_call", "d_call", "j_call",
                  "junction", "junction_aa", "productive"}
_CHANGEO_FIELDS = {"sequence_id", "sequence", "v_call", "d_call", "j_call",
                   "junction", "junction_length", "sequence_alignment", "germline_alignment",
                   "np1", "np2"}


def _complete_record():
    seq = "GAGGTGCAGCTGGTGGAGTCTGGGGGAGGC"
    return {
        "sequence": seq, "orientation_id": 0, "locus": "IGH",
        "productive": True, "vj_in_frame": True, "stop_codon": False,
        "v_call": "IGHV3-23*01", "d_call": "IGHD3-10*01", "j_call": "IGHJ4*02",
        "v_sequence_start": 0, "v_sequence_end": 20, "v_germline_start": 0, "v_germline_end": 20,
        "d_sequence_start": 20, "d_sequence_end": 24, "d_germline_start": 0, "d_germline_end": 4,
        "j_sequence_start": 24, "j_sequence_end": 30, "j_germline_start": 0, "j_germline_end": 6,
        "v_cigar": "20M", "d_cigar": "20S4M6S", "j_cigar": "24S6M",
        "junction": "TGTGCGAGAGGG", "junction_aa": "CARG", "junction_length": 12,
        "sequence_alignment": seq, "germline_alignment": seq, "v_identity": 0.99,
    }


@pytest.mark.parametrize("name,required", [
    ("MiAIRR/airrflow", _MIAIRR_REQUIRED),
    ("Scirpy", _SCIRPY_FIELDS),
    ("Change-O", _CHANGEO_FIELDS),
])
def test_output_columns_cover_downstream_importers(name, required):
    missing = required - set(COLUMNS)
    assert not missing, f"{name} reads columns AlignAIR does not emit: {sorted(missing)}"


def test_emitted_tsv_validates_against_airr_schema():
    tmp = tempfile.mktemp(suffix=".tsv")
    try:
        rec = _complete_record()
        write_airr(tmp, ["read1"], [rec["sequence"]], [rec], locus="IGH")
        assert airr.validate_rearrangement(tmp), "emitted TSV failed airr.validate_rearrangement"
        # and it reads back through the official reader
        rows = list(airr.read_rearrangement(tmp))
        assert rows and rows[0]["sequence_id"] == "read1"
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
