"""FWR/CDR region extraction and the junction of the predicted anchors.

A model's V and J anchors give its junction in read coordinates (``predict.junction.derive_junction``);
:func:`anchor_junction_fields` gives the AIRR fields of it. The CDR3 positions are read coordinates, 0-based and
end-exclusive in the record; the AIRR writer emits them 1-based and closed.
"""
from __future__ import annotations

from typing import Optional

from .translate import translate_nt
from .constants import IMGT_REGIONS

_FWR_CDR = ("fwr1", "cdr1", "fwr2", "cdr2", "fwr3")


def extract_regions(seq_alignment: Optional[str], seq_alignment_aa: Optional[str]) -> dict:
    """The FWR1-FWR3 and CDR1-CDR2 fields (sequence, amino acids, start, end) cut from the IMGT-gapped read alignment
    at the IMGT region columns; all None without an alignment."""
    result: dict = {}
    if seq_alignment is None:
        for region in _FWR_CDR:
            result[region] = result[f"{region}_aa"] = None
            result[f"{region}_start"] = result[f"{region}_end"] = None
        return result
    for region in _FWR_CDR:
        start, end = IMGT_REGIONS[region]
        result[region] = seq_alignment[start:end]
        result[f"{region}_start"], result[f"{region}_end"] = start, end
        result[f"{region}_aa"] = (seq_alignment_aa[start // 3:end // 3]
                                  if seq_alignment_aa is not None else None)
    return result


# The fields a read-coordinate junction sets (``read_junction_fields``).
READ_JUNCTION_KEYS = ("junction", "junction_aa", "junction_length", "junction_aa_length", "junction_start",
                      "junction_end", "cdr3", "cdr3_aa", "cdr3_start", "cdr3_end", "fwr4", "fwr4_aa", "fwr4_start",
                      "fwr4_end")


def read_junction_fields(seq: str, js: int, je: int, j_seq_end: Optional[int], translate_fn) -> dict:
    """The junction ``seq[js:je]`` (read coordinates, 0-based, end-exclusive) and what follows from it: its
    translation (``translate_fn``), the CDR3 (the junction without its two anchor codons) and FWR4 (from the
    junction's end to the J end ``j_seq_end``), all in read coordinates."""
    junction_nt = seq[js:je].upper()
    junction_aa = translate_fn(junction_nt)
    cdr3_start, cdr3_end = js + 3, je - 3
    fwr4 = seq[je:j_seq_end].upper() if (j_seq_end is not None and j_seq_end >= je) else ""
    return {
        "junction": junction_nt, "junction_aa": junction_aa,
        "junction_length": len(junction_nt), "junction_aa_length": len(junction_aa),
        "junction_start": js, "junction_end": je,
        "cdr3": seq[cdr3_start:cdr3_end].upper(),
        "cdr3_aa": junction_aa[1:-1] if len(junction_aa) > 2 else "",
        "cdr3_start": cdr3_start, "cdr3_end": cdr3_end,
        "fwr4": fwr4, "fwr4_aa": translate_fn(fwr4) if fwr4 else "",
        "fwr4_start": je, "fwr4_end": j_seq_end,
    }


def anchor_junction_fields(seq: str, call, j_seq_end: Optional[int]) -> dict:
    """The AIRR junction fields of a junction derived from the predicted anchors (``predict.junction.JunctionCall``):
    ``read_junction_fields`` with GenAIRR's translation convention (full codons from the junction's start, a partial
    last codon dropped, X for a codon with an N), and the frame, stop codon and productivity of the call. Every field
    is None when the junction is unknown."""
    flags = {"vj_in_frame": call.vj_in_frame, "stop_codon": call.stop_codon, "productive": call.productive}
    if call.junction_start is None:
        return {**dict.fromkeys(READ_JUNCTION_KEYS), **flags}
    return {**read_junction_fields(seq, call.junction_start, call.junction_end, j_seq_end, translate_nt), **flags}
