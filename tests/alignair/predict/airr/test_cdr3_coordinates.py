"""The CDR3 positions of an AIRR row are positions in the read (the AIRR schema's "query sequence"), on both junction
paths: the record holds them 0-based and end-exclusive, like its other read coordinates, and the writer emits them
1-based and closed (``cdr3_start`` + 1; the end-exclusive end is already the closed end). The CDR3 is the junction
without its two conserved codons."""
from types import SimpleNamespace

import pytest

from alignair.io.airr import _build_row
from alignair.predict.airr import build_airr
from alignair.reference.codons import IMGT_CODON_104

pytestmark = pytest.mark.component

CYS_COLUMN = IMGT_CODON_104                                    # 309
GAPS = 6                                                       # IMGT gap dots at columns 30..35
CYS = CYS_COLUMN - GAPS                                        # ungapped Cys position: 303
V_GERM = ("GCA" * 101)[:CYS] + "TGT" + "GCGAGAGAC"[:315 - CYS - 3]   # 315 germline bases, the Cys at 303
V_GAPPED = V_GERM[:30] + "." * GAPS + V_GERM[30:]
J_GERM, J_ANCHOR = "ACTACTTTGA" + "TGG" + "GGCCAGGGAACC", 10
V_GERM_START = 5                                               # the read starts 5 bases into the V germline
NP = "GGGAAA"


def _reference(v_gapped=None):
    v_gapped = {"V1*01": V_GAPPED} if v_gapped is None else v_gapped
    genes = {"V": SimpleNamespace(names=["V1*01"], sequences=[V_GERM], gapped=v_gapped, anchors={}),
             "J": SimpleNamespace(names=["J1*01"], sequences=[J_GERM], gapped=None, anchors={"J1*01": J_ANCHOR})}
    return SimpleNamespace(genes=set(genes), gene=genes.__getitem__, loci=[])


def _record(v_cigar=None, v_part=None):
    """A light read of V1 and J1, typed, its anchors the V Cys and the J Trp codon starts in the read."""
    from alignair.predict.results import Anchors
    from alignair.predict.state import GermlineAlignment
    from tests.support.read_results import make_read
    v_part = V_GERM[V_GERM_START:] if v_part is None else v_part
    read = v_part + NP + J_GERM
    j_start = len(v_part) + len(NP)
    v_anchor = CYS - V_GERM_START - (len(V_GERM) - V_GERM_START - len(v_part))
    return make_read(read, calls={"v": "V1*01", "j": "J1*01"}, anchors=Anchors(v_anchor, j_start + J_ANCHOR),
                     alignments={"v": GermlineAlignment("V1*01", 0, len(v_part), V_GERM_START, len(V_GERM),
                                                        v_cigar or f"{len(v_part)}M"),
                                 "j": GermlineAlignment("J1*01", j_start, len(read), 0, len(J_GERM),
                                                        f"{len(J_GERM)}M")})


def test_the_anchors_put_the_junction_and_cdr3_in_read_coordinates():
    """The junction runs from the V Cys codon to the end of the J Trp codon, in the read; the CDR3 is the junction
    without those two codons."""
    record = _record()
    row = build_airr([record], _reference(), has_d=False)[0]
    read, cys = record.sequence, CYS - V_GERM_START
    junction_end = record.genes["j"].alignment.seq_start + J_ANCHOR + 3
    assert (row["junction_start"], row["junction_end"]) == (cys, junction_end)
    assert row["junction"] == read[cys:junction_end]
    assert (row["cdr3_start"], row["cdr3_end"]) == (cys + 3, junction_end - 3)
    assert row["cdr3"] == read[row["cdr3_start"]:row["cdr3_end"]] == row["junction"][3:-3]


def test_a_v_deletion_before_the_cys_moves_the_junction_with_the_anchor():
    v_part = V_GERM[V_GERM_START:100] + V_GERM[101:]               # one V germline base deleted
    record = _record(v_cigar=f"{100 - V_GERM_START}M1D{len(V_GERM) - 101}M", v_part=v_part)
    row = build_airr([record], _reference(), has_d=False)[0]
    read, cys = record.sequence, CYS - V_GERM_START - 1
    assert (row["junction_start"], row["cdr3_start"], row["cdr3_end"]) == (cys, cys + 3, row["junction_end"] - 3)
    assert row["cdr3"] == read[row["cdr3_start"]:row["cdr3_end"]] == row["junction"][3:-3]


@pytest.mark.parametrize("indel", [False, True], ids=["no_indel", "v_deletion"])
def test_a_written_row_gives_the_cdr3_as_1_based_closed_positions_in_the_read(indel):
    """The AIRR schema: ``cdr3_start``/``cdr3_end`` are the CDR3's start and end positions in the query sequence,
    1-based, closed; the junction is the CDR3 plus the two conserved codons around it."""
    v_part = V_GERM[V_GERM_START:100] + V_GERM[101:] if indel else None
    record = _record(v_cigar=f"{100 - V_GERM_START}M1D{len(V_GERM) - 101}M" if indel else None, v_part=v_part)
    row = build_airr([record], _reference(), has_d=False)[0]
    written = _build_row("r", record.sequence, row, "IGH")
    start, end, query = written["cdr3_start"], written["cdr3_end"], written["sequence"]
    assert (start, end) == (row["cdr3_start"] + 1, row["cdr3_end"])
    assert query[start - 1:end] == written["cdr3"]
    assert query[start - 1 - 3:end + 3] == written["junction"]


def test_a_row_without_a_cdr3_writes_none():
    written = _build_row("r", "ACGT", {"sequence": "ACGT", "cdr3_start": None, "cdr3_end": None}, "IGH")
    assert written["cdr3_start"] is None and written["cdr3_end"] is None


def test_fwr4_is_imgts_fr4_from_the_j_trp_codon():
    """IMGT's FR4 starts at the J Trp (or Phe) codon, IMGT position 118, the junction's last codon, and runs to the
    J's end, as IgBLAST and Change-O's MakeDb write it: the codon after the CDR3. Written, it starts one past
    ``cdr3_end``."""
    record = _record()
    row = build_airr([record], _reference(), has_d=False)[0]
    read, junction_end = record.sequence, row["junction_end"]
    assert (row["fwr4_start"], row["fwr4_end"]) == (junction_end - 3, len(read))
    assert row["fwr4"] == read[junction_end - 3:] and row["fwr4"][:3] == row["junction"][-3:] == "TGG"
    assert row["fwr4_aa"].startswith("WGQG")
    written = _build_row("r", read, row, "IGH")
    assert written["fwr4_start"] == written["cdr3_end"] + 1
    assert written["sequence"][written["fwr4_start"] - 1:written["fwr4_end"]] == written["fwr4"]


def test_a_v_without_a_gapped_form_has_no_alignment_positions():
    """A called V allele the reference holds no IMGT-gapped form of (a bound reference that gaps its other alleles)
    gets no gapped alignment: the row is partial, and it writes no alignment positions and no V germline positions,
    which count in the IMGT-gapped germline. Its v_cigar still places the V in the ungapped germline."""
    record = _record()
    row = build_airr([record], _reference(v_gapped={"V2*01": V_GAPPED}), has_d=False)[0]
    assert row["airr_assembly_status"] == "partial" and not row.get("sequence_alignment")
    written = _build_row("r", record.sequence, row, "IGH")
    blank = ("v_germline_start", "v_germline_end", "v_alignment_start", "v_alignment_end", "j_alignment_start",
             "j_alignment_end")
    assert {key: written[key] for key in blank} == dict.fromkeys(blank)
    assert written["v_cigar"].startswith(f"{V_GERM_START}N") and written["j_germline_start"] == 1
