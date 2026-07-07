"""AIRR quality flags: stop_codon, vj_in_frame, v_identity (faithful port of TF AIRR/quality)."""
from __future__ import annotations

from typing import Optional


def stop_codon(seq_alignment_aa: Optional[str]) -> Optional[bool]:
    return ("*" in seq_alignment_aa) if seq_alignment_aa is not None else None


def vj_in_frame(cdr3_start, cdr3_end, v_alignment_start) -> Optional[bool]:
    if cdr3_start is None or cdr3_end is None or v_alignment_start is None:
        return None
    return (cdr3_end - v_alignment_start) % 3 == 0 and (cdr3_end - cdr3_start) % 3 == 0


def v_identity(v_sequence_alignment: Optional[str], v_germline_alignment: Optional[str]) -> Optional[float]:
    """Fraction of matching nucleotides over non-gap germline positions."""
    if not v_sequence_alignment or not v_germline_alignment:
        return None
    matches = compared = 0
    for s, g in zip(v_sequence_alignment, v_germline_alignment):
        if g == ".":
            continue
        compared += 1
        if s == g:
            matches += 1
    return matches / compared if compared else None
