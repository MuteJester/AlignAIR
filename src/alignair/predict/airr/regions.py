"""FWR/CDR region extraction + junction/CDR3 via the J anchor.

Two junction derivations live here. :func:`compute_junction` slices the fixed IMGT column 309 out of
the (indel-blind) gapped ``sequence_alignment`` — exact on reads without a V/J-region indel.
:func:`compute_junction_cigar` re-derives the junction in *read* coordinates by mapping the conserved
2nd-Cys (column 309) and the J anchor through the reader's CIGAR, so a V/J indel between the V start
and the junction no longer shifts the anchor. The builder uses the CIGAR path only when the read
carries an indel (see :func:`cigar_has_indel`), leaving indel-free reads byte-identical.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

from GenAIRR.utilities import translate

from .constants import IMGT_REGIONS

_CIGAR = re.compile(r"(\d+)([MIDNS=X])")
_JUNCTION_COL = IMGT_REGIONS["junction"][0]    # gapped IMGT column of the conserved 2nd-Cys (309)

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


def cigar_has_indel(cigar: Optional[str]) -> bool:
    """True iff the CIGAR contains an insertion or deletion op (I/D)."""
    return any(op in "ID" for _, op in _CIGAR.findall(cigar or ""))


def cys_position(v_ref_gapped: str) -> Optional[int]:
    """Ungapped V-germline position of the conserved 2nd-Cys (IMGT gapped column 309).

    The gapped reference places the Cys at column 309; subtracting the IMGT gap dots before that
    column converts it to an ungapped germline coordinate. Returns None when the allele's gapped
    reference does not reach the column (e.g. a partial/absent reference)."""
    if not v_ref_gapped or len(v_ref_gapped) <= _JUNCTION_COL:
        return None
    return _JUNCTION_COL - v_ref_gapped[:_JUNCTION_COL].count(".")


def map_germline_to_read(g_target: int, germ_start: int, seq_start: int, seq_end: int,
                         cigar: Optional[str]) -> int:
    """Read position aligned to ungapped-germline position ``g_target``, walking the CIGAR (which
    describes ``read[seq_start:seq_end]`` against ``germline[germ_start:germ_end]``). Insertions
    advance only the read, deletions only the germline. A ``g_target`` past the aligned germline
    (e.g. a 3'-trimmed V whose Cys is gone) clamps to ``seq_end``."""
    if g_target < germ_start:
        return seq_start
    r, g = seq_start, germ_start
    for count, op in _CIGAR.findall(cigar or ""):
        for _ in range(int(count)):
            if g == g_target:
                return r
            if op in "M=X":
                r += 1
                g += 1
            elif op == "D":
                g += 1
            elif op in "IS":
                r += 1
            if g > g_target:
                return r
    return seq_end


def compute_junction_cigar(seq, v_ref_gapped, v_germ_start, v_seq_start, v_seq_end, v_cigar,
                           j_anchor, j_germ_start, j_seq_start, j_seq_end, j_cigar) -> Optional[dict]:
    """Indel-robust junction in *read* coordinates: the 5' bound is the conserved 2nd-Cys mapped
    through the V CIGAR, the 3' bound is the J anchor (Trp/Phe) mapped through the J CIGAR, ``+3``.
    Returns the same keys as :func:`compute_junction` (plus read-coordinate ``junction_start`` /
    ``junction_end``), or None if the anchors cannot be placed (caller falls back to the column path).
    """
    cys = cys_position(v_ref_gapped)
    if cys is None or v_seq_start is None or j_seq_start is None:
        return None
    js = map_germline_to_read(cys, v_germ_start or 0, v_seq_start, v_seq_end, v_cigar)
    je = map_germline_to_read(j_anchor, j_germ_start or 0, j_seq_start, j_seq_end, j_cigar) + 3
    if je <= js or je > len(seq):
        return None
    junction_nt = seq[js:je].upper()
    junction_aa = translate(junction_nt)
    cdr3_start, cdr3_end = js + 3, je - 3
    fwr4 = seq[je:j_seq_end].upper() if (j_seq_end is not None and j_seq_end >= je) else ""
    return {
        "junction": junction_nt, "junction_aa": junction_aa,
        "junction_length": len(junction_nt), "junction_aa_length": len(junction_aa),
        "junction_start": js, "junction_end": je,
        "cdr3": seq[cdr3_start:cdr3_end].upper(),
        "cdr3_aa": junction_aa[1:-1] if len(junction_aa) > 2 else "",
        "cdr3_start": cdr3_start, "cdr3_end": cdr3_end,
        "fwr4": fwr4, "fwr4_aa": translate(fwr4) if fwr4 else "",
        "fwr4_start": je, "fwr4_end": j_seq_end,
    }
