"""The CDR3 positions of an AIRR row are positions in the read (the AIRR schema's "query sequence"), on both junction
paths: the record holds them 0-based and end-exclusive, like its other read coordinates, and the writer emits them
1-based and closed (``cdr3_start`` + 1; the end-exclusive end is already the closed end). The CDR3 is the junction
without its two conserved codons."""
from types import SimpleNamespace

import pytest

from alignair.io.airr import _build_row
from alignair.predict.airr import build_airr
from alignair.predict.airr.regions import IMGT_REGIONS, read_position

pytestmark = pytest.mark.component

CYS_COLUMN = IMGT_REGIONS["junction"][0]                       # 309
GAPS = 6                                                       # IMGT gap dots at columns 30..35
CYS = CYS_COLUMN - GAPS                                        # ungapped Cys position: 303
V_GERM = ("GCA" * 101)[:CYS] + "TGT" + "GCGAGAGAC"[:315 - CYS - 3]   # 315 germline bases, the Cys at 303
V_GAPPED = V_GERM[:30] + "." * GAPS + V_GERM[30:]
J_GERM, J_ANCHOR = "ACTACTTTGA" + "TGG" + "GGCCAGGGAACC", 10
V_GERM_START = 5                                               # the read starts 5 bases into the V germline
NP = "GGGAAA"


def _reference():
    genes = {"V": SimpleNamespace(names=["V1*01"], sequences=[V_GERM], gapped={"V1*01": V_GAPPED}, anchors={}),
             "J": SimpleNamespace(names=["J1*01"], sequences=[J_GERM], gapped=None, anchors={"J1*01": J_ANCHOR})}
    return SimpleNamespace(genes=set(genes), gene=genes.__getitem__, loci=[])


def _record(v_cigar=None, v_part=None):
    v_part = V_GERM[V_GERM_START:] if v_part is None else v_part
    read = v_part + NP + J_GERM
    j_start = len(v_part) + len(NP)
    return {"sequence_id": "r", "sequence": read, "input_sequence": read, "orientation_id": 0,
            "v_call": "V1*01", "j_call": "J1*01", "productive": True, "indel_count": 0,
            "v_sequence_start": 0, "v_sequence_end": len(v_part),
            "v_germline_start": V_GERM_START, "v_germline_end": len(V_GERM), "v_cigar": v_cigar or f"{len(v_part)}M",
            "j_sequence_start": j_start, "j_sequence_end": len(read),
            "j_germline_start": 0, "j_germline_end": len(J_GERM), "j_cigar": f"{len(J_GERM)}M"}


def test_a_gapped_column_maps_to_the_read_base_at_or_after_it():
    alignment = "..AC.GT"                                      # 2 germline bases the read lacks, an IMGT gap at 4
    assert [read_position(alignment, 10, c) for c in range(8)] == [10, 10, 10, 11, 12, 12, 13, 14]


def test_the_column_path_puts_the_cdr3_in_read_coordinates():
    """A read without an indel takes its junction from the gapped alignment's IMGT columns; its CDR3 positions are
    the read's (here 5 germline bases the read lacks and 6 IMGT gaps lie before the Cys column)."""
    record = _record()
    row = build_airr([record], _reference(), chain="light")[0]
    assert row["airr_assembly_status"] == "complete" and "junction_start" not in row      # the column path
    read, cys = record["sequence"], CYS - V_GERM_START
    junction_end = record["j_sequence_start"] + J_ANCHOR + 3
    assert row["junction"] == read[cys:junction_end]
    assert (row["cdr3_start"], row["cdr3_end"]) == (cys + 3, junction_end - 3)
    assert row["cdr3"] == read[row["cdr3_start"]:row["cdr3_end"]] == row["junction"][3:-3]


def test_the_cigar_path_puts_the_cdr3_in_read_coordinates():
    """A read with a V indel takes its junction through the CIGAR, in read coordinates."""
    v_part = V_GERM[V_GERM_START:100] + V_GERM[101:]               # one V germline base deleted
    record = _record(v_cigar=f"{100 - V_GERM_START}M1D{len(V_GERM) - 101}M", v_part=v_part)
    row = build_airr([record], _reference(), chain="light")[0]
    read, cys = record["sequence"], CYS - V_GERM_START - 1
    assert (row["junction_start"], row["cdr3_start"], row["cdr3_end"]) == (cys, cys + 3, row["junction_end"] - 3)
    assert row["cdr3"] == read[row["cdr3_start"]:row["cdr3_end"]] == row["junction"][3:-3]


@pytest.mark.parametrize("indel", [False, True], ids=["column_path", "cigar_path"])
def test_a_written_row_gives_the_cdr3_as_1_based_closed_positions_in_the_read(indel):
    """The AIRR schema: ``cdr3_start``/``cdr3_end`` are the CDR3's start and end positions in the query sequence,
    1-based, closed; the junction is the CDR3 plus the two conserved codons around it."""
    v_part = V_GERM[V_GERM_START:100] + V_GERM[101:] if indel else None
    record = _record(v_cigar=f"{100 - V_GERM_START}M1D{len(V_GERM) - 101}M" if indel else None, v_part=v_part)
    row = build_airr([record], _reference(), chain="light")[0]
    written = _build_row("r", record["sequence"], row, "IGH")
    start, end, query = written["cdr3_start"], written["cdr3_end"], written["sequence"]
    assert (start, end) == (row["cdr3_start"] + 1, row["cdr3_end"])
    assert query[start - 1:end] == written["cdr3"]
    assert query[start - 1 - 3:end + 3] == written["junction"]


def test_a_row_without_a_cdr3_writes_none():
    written = _build_row("r", "ACGT", {"sequence": "ACGT", "cdr3_start": None, "cdr3_end": None}, "IGH")
    assert written["cdr3_start"] is None and written["cdr3_end"] is None
