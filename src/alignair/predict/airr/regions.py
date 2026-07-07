"""FWR/CDR region extraction + junction/CDR3 via the J anchor (faithful port of TF AIRR/regions)."""
from __future__ import annotations

from typing import Dict, Optional

from .constants import IMGT_REGIONS

_FWR_CDR = ("fwr1", "cdr1", "fwr2", "cdr2", "fwr3")


def extract_regions(seq_alignment: Optional[str], seq_alignment_aa: Optional[str]) -> dict:
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


def compute_junction(seq_alignment, seq_alignment_aa, j_call, j_anchor_dict,
                     j_seq_start, v_seq_start, j_germ_start, j_alignment_end) -> dict:
    """Junction end from the conserved J anchor:
    ``junc_end = j_seq_start - v_seq_start + gap_count + j_anchor - j_germ_start + 3``."""
    keys = ("junction", "junction_aa", "junction_length", "junction_aa_length",
            "cdr3", "cdr3_aa", "cdr3_start", "cdr3_end", "fwr4", "fwr4_aa", "fwr4_start", "fwr4_end")
    if seq_alignment is None or seq_alignment_aa is None:
        return {k: None for k in keys}

    j_anchor = (j_anchor_dict or {}).get(j_call.split(",")[0], 0)
    offset = seq_alignment.count(".")
    junction_start = IMGT_REGIONS["junction"][0]
    junc_end = j_seq_start - v_seq_start + offset + j_anchor - j_germ_start + 3

    junction_nt = seq_alignment[junction_start:junc_end]
    junction_aa = seq_alignment_aa[junction_start // 3:junc_end // 3]
    cdr3_start, cdr3_end = junction_start + 3, junc_end - 3
    fwr4_start = cdr3_end + 1
    return {
        "junction": junction_nt, "junction_aa": junction_aa,
        "junction_length": len(junction_nt), "junction_aa_length": len(junction_aa),
        "cdr3": seq_alignment[cdr3_start:cdr3_end],
        "cdr3_aa": junction_aa[1:-1] if len(junction_aa) > 2 else "",
        "cdr3_start": cdr3_start, "cdr3_end": cdr3_end,
        "fwr4": seq_alignment[fwr4_start:j_alignment_end] if j_alignment_end is not None else None,
        "fwr4_aa": (seq_alignment_aa[fwr4_start // 3:j_alignment_end // 3]
                    if j_alignment_end is not None else None),
        "fwr4_start": fwr4_start, "fwr4_end": j_alignment_end,
    }
