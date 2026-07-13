"""P0-16 layer 4 — a golden AIRR output (fixed inputs -> exact normalized rows) that also passes the
**official** `airr` Python library's schema validation, so our emitted rows are standards-conformant."""
import csv

import pytest

from alignair.io.airr import write_airr

airr = pytest.importorskip("airr")

# Fixed, fully-assembled records (as build_airr would produce), coordinates 0-based end-exclusive.
_RECORDS = [
    {"sequence": "ACGTACGTACGTACGT", "orientation_id": 0, "productive": True, "vj_in_frame": True,
     "stop_codon": False, "v_call": "IGHV1-2*02", "d_call": "IGHD3-3*01", "j_call": "IGHJ4*02",
     "v_sequence_start": 0, "v_sequence_end": 8, "v_germline_start": 0, "v_germline_end": 8, "v_cigar": "8M8S",
     "d_sequence_start": 8, "d_sequence_end": 11, "d_germline_start": 0, "d_germline_end": 3, "d_cigar": "8S3M5S",
     "j_sequence_start": 11, "j_sequence_end": 16, "j_germline_start": 0, "j_germline_end": 5, "j_cigar": "11S5M",
     "junction": "TGTGCGAGA", "junction_aa": "CAR", "junction_length": 9,
     "sequence_alignment": "ACGTACGTACGTACGT", "germline_alignment": "ACGTACGTACGTACGT"},
    {"sequence": "TTTTGGGGCCCCAAAA", "orientation_id": 1, "productive": False, "vj_in_frame": False,
     "stop_codon": True, "v_call": "IGHV3-23*01", "d_call": "", "j_call": "IGHJ6*02",
     "v_sequence_start": 0, "v_sequence_end": 10, "v_germline_start": 0, "v_germline_end": 10, "v_cigar": "10M6S",
     "j_sequence_start": 10, "j_sequence_end": 16, "j_germline_start": 0, "j_germline_end": 6, "j_cigar": "10S6M",
     "junction": "TGTGCG", "junction_aa": "CA", "junction_length": 6,
     "sequence_alignment": "TTTTGGGGCCCCAAAA", "germline_alignment": "TTTTGGGGCCCCAAAA"},
]


def test_golden_airr_rows_are_exact():
    import tempfile
    import os
    out = tempfile.mktemp(suffix=".tsv")
    try:
        write_airr(out, ["r0", "r1"], [r["sequence"] for r in _RECORDS], _RECORDS, locus="IGH")
        rows = list(csv.DictReader(open(out), delimiter="\t"))
    finally:
        os.remove(out)
    r0, r1 = rows
    assert r0["v_call"] == "IGHV1-2*02" and r0["rev_comp"] == "F" and r0["productive"] == "T"
    assert r0["v_sequence_start"] == "1" and r0["v_sequence_end"] == "8"   # 0-based 0..8 -> 1-based 1..8
    assert r0["junction"] == "TGTGCGAGA" and r0["locus"] == "IGH"
    assert r1["rev_comp"] == "T" and r1["productive"] == "F" and r1["stop_codon"] == "T"


def test_golden_airr_passes_official_validation(tmp_path):
    out = str(tmp_path / "golden.tsv")
    write_airr(out, ["r0", "r1"], [r["sequence"] for r in _RECORDS], _RECORDS,
               locus="IGH", columns="airr")
    # official airr validation raises airr.ValidationError on a non-conformant row
    rows = list(airr.read_rearrangement(out, validate=True))
    assert len(rows) == 2
    assert rows[0]["v_call"] == "IGHV1-2*02"
