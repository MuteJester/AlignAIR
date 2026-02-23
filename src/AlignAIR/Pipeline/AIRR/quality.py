"""Quality and productivity flags for AIRR rearrangement records."""
from __future__ import annotations

from typing import Optional


def has_stop_codon(seq_alignment_aa: Optional[str]) -> bool:
    """Check whether the translated alignment contains a stop codon ('*')."""
    if seq_alignment_aa is None:
        return False
    return '*' in seq_alignment_aa


def is_vj_in_frame(
    cdr3_start: Optional[int],
    cdr3_end: Optional[int],
    v_alignment_start: Optional[int],
) -> bool:
    """Check whether the V-J junction is in-frame.

    The rearrangement is in-frame when:
    - (cdr3_end - v_alignment_start) % 3 == 0
    - (cdr3_end - cdr3_start) % 3 == 0
    """
    if cdr3_start is None or cdr3_end is None or v_alignment_start is None:
        return False
    return (
        (cdr3_end - v_alignment_start) % 3 == 0
        and (cdr3_end - cdr3_start) % 3 == 0
    )


def compute_v_identity(
    v_seq_alignment: Optional[str],
    v_germ_alignment: Optional[str],
) -> Optional[float]:
    """Compute nucleotide identity between V sequence and germline alignment.

    Skips gap positions ('.') in the germline.
    Returns a fraction in [0.0, 1.0] or None if inputs are invalid.
    """
    if not v_seq_alignment or not v_germ_alignment:
        return None

    matches = 0
    total = 0
    for s, g in zip(v_seq_alignment, v_germ_alignment):
        if g == '.':
            continue
        total += 1
        if s == g:
            matches += 1

    if total == 0:
        return None
    return matches / total
