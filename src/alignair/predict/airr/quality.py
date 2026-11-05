"""Segment identity of an AIRR row's sequence/germline alignment."""
from __future__ import annotations

from typing import Optional


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

