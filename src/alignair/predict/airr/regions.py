"""FWR/CDR region extraction and the junction of the predicted anchors.

A model's V and J anchors give its junction in read coordinates (``predict.junction.derive_junction``);
:func:`anchor_junction_fields` gives the AIRR fields of it. Every region's positions (FWR1-FWR4, CDR1-CDR3) are read
coordinates, 0-based and end-exclusive in the record; the AIRR writer emits them 1-based and closed.
"""
from __future__ import annotations

from .constants import IMGT_REGIONS
from .translate import translate_nt

_FWR_CDR = ("fwr1", "cdr1", "fwr2", "cdr2", "fwr3")


def extract_regions(seq_alignment: str | None, seq_alignment_aa: str | None,
                    read_positions: list[int | None] | None) -> dict:
    """The FWR1-FWR3 and CDR1-CDR2 fields cut from the IMGT-gapped read alignment at the IMGT region columns: the
    region's bases and amino acids as the alignment holds them (gaps included), and its start and end in the read
    (0-based, end-exclusive): from the first to the last read base in its columns (``read_positions``, one per
    alignment column, ``alignment.sequence_alignment_columns``), so a base the alignment leaves out (a V insertion)
    stays inside them. A region with no read base has no start or end; all None without an alignment."""
    result: dict = {}
    if seq_alignment is None:
        for region in _FWR_CDR:
            result[region] = result[f"{region}_aa"] = None
            result[f"{region}_start"] = result[f"{region}_end"] = None
        return result
    for region in _FWR_CDR:
        start, end = IMGT_REGIONS[region]
        result[region] = seq_alignment[start:end]
        covered = [p for p in (read_positions or [])[start:end] if p is not None]
        span = (min(covered), max(covered) + 1) if covered else (None, None)
        result[f"{region}_start"], result[f"{region}_end"] = span
        result[f"{region}_aa"] = (seq_alignment_aa[start // 3:end // 3]
                                  if seq_alignment_aa is not None else None)
    return result


# The fields a read-coordinate junction sets (``read_junction_fields``).
READ_JUNCTION_KEYS = ("junction", "junction_aa", "junction_length", "junction_aa_length", "junction_start",
                      "junction_end", "cdr3", "cdr3_aa", "cdr3_start", "cdr3_end", "fwr4", "fwr4_aa", "fwr4_start",
                      "fwr4_end")


def read_junction_fields(seq: str, js: int, je: int, j_seq_end: int | None, translate_fn) -> dict:
    """The junction ``seq[js:je]`` (read coordinates, 0-based, end-exclusive) and what follows from it: its
    translation (``translate_fn``), the CDR3 (the junction without its two anchor codons) and FWR4, IMGT's FR4: from
    the junction's last codon (the J Trp or Phe, IMGT position 118) to the J end ``j_seq_end``, as IgBLAST and
    Change-O's MakeDb cut it; none when the J ends inside that codon. All in read coordinates."""
    junction_nt = seq[js:je].upper()
    junction_aa = translate_fn(junction_nt)
    cdr3_start, cdr3_end = js + 3, je - 3
    fwr4 = seq[je - 3:j_seq_end].upper() if (j_seq_end is not None and j_seq_end >= je) else ""
    fwr4_start, fwr4_end = (je - 3, j_seq_end) if fwr4 else (None, None)
    return {
        "junction": junction_nt, "junction_aa": junction_aa,
        "junction_length": len(junction_nt), "junction_aa_length": len(junction_aa),
        "junction_start": js, "junction_end": je,
        "cdr3": seq[cdr3_start:cdr3_end].upper(),
        "cdr3_aa": junction_aa[1:-1] if len(junction_aa) > 2 else "",
        "cdr3_start": cdr3_start, "cdr3_end": cdr3_end,
        "fwr4": fwr4, "fwr4_aa": translate_fn(fwr4) if fwr4 else "",
        "fwr4_start": fwr4_start, "fwr4_end": fwr4_end,
    }


def anchor_junction_fields(seq: str, call, j_seq_end: int | None) -> dict:
    """The AIRR junction fields of a junction derived from the predicted anchors (``predict.junction.JunctionCall``):
    ``read_junction_fields`` with GenAIRR's translation convention (full codons from the junction's start, a partial
    last codon dropped, X for a codon with an N), and the frame, stop codon and productivity of the call. Every field
    is None when the junction is unknown."""
    flags = {"vj_in_frame": call.vj_in_frame, "stop_codon": call.stop_codon, "productive": call.productive}
    if call.junction_start is None:
        return {**dict.fromkeys(READ_JUNCTION_KEYS), **flags}
    return {**read_junction_fields(seq, call.junction_start, call.junction_end, j_seq_end, translate_nt), **flags}
