"""AIRR quality flags: stop_codon, vj_in_frame, v_identity."""
from __future__ import annotations

from typing import Optional


def stop_codon(seq_alignment_aa: Optional[str]) -> Optional[bool]:
    return ("*" in seq_alignment_aa) if seq_alignment_aa is not None else None


def segment_identity(seq_alignment: Optional[str], germ_alignment: Optional[str]) -> Optional[float]:
    """Fraction of matching nucleotides over the germline positions the read covers (any V/D/J segment): an IMGT gap
    column and germline the read does not reach (``.`` in either string) are skipped; a germline base the read lacks
    (``-``) is a mismatch."""
    if not seq_alignment or not germ_alignment:
        return None
    matches = compared = 0
    for s, g in zip(seq_alignment, germ_alignment):
        if g == "." or s == ".":
            continue
        compared += 1
        if s == g:
            matches += 1
    return matches / compared if compared else None


def v_identity(v_sequence_alignment: Optional[str], v_germline_alignment: Optional[str]) -> Optional[float]:
    """Back-compat alias of :func:`segment_identity` for the V segment."""
    return segment_identity(v_sequence_alignment, v_germline_alignment)
