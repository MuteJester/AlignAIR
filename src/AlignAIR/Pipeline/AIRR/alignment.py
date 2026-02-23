"""IMGT-gapped alignment construction for AIRR rearrangement records.

All public functions operate on a single sequence (no batch loops).
The builder module handles iteration and error wrapping.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from GenAIRR.utilities import translate

from AlignAIR.Pipeline.AIRR.constants import SHORT_D_SENTINEL
from AlignAIR.Pipeline.AIRR.references import ReferenceData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_gaps(ref: str) -> int:
    """Count IMGT gap characters ('.') in a reference sequence."""
    return ref.count('.')


def _get_reference_segment(ref_seq: str, end: int, start: int = 0) -> str:
    """Slice a reference sequence, adjusting end for IMGT gaps."""
    if not ref_seq:
        return ''
    gap_adjusted_end = end + _count_gaps(ref_seq)
    return ref_seq[start:gap_adjusted_end] if start > 0 else ref_seq[:gap_adjusted_end]


# ---------------------------------------------------------------------------
# Sequence alignment (IMGT-gapped)
# ---------------------------------------------------------------------------

def build_sequence_alignment(
    seq: str,
    v_ref_gapped: str,
    v_seq_start: int,
    v_seq_end: int,
    v_germ_start: int,
    v_germ_end: int,
    j_seq_end: int,
) -> Optional[str]:
    """Build the IMGT-gapped sequence alignment string.

    Inserts IMGT gap characters ('.') into the sequence to match the
    V-germline gapped reference, then appends the D+J portion as-is.
    """
    if not v_ref_gapped:
        return None

    gaps = _count_gaps(v_ref_gapped)
    v_ref_trimmed = v_ref_gapped[:v_germ_end + gaps]

    # The portion of the query that aligns to V
    seq_to_gap = seq[v_seq_start:v_seq_end]

    # The remainder (D + J region, unaligned)
    seq_remain = seq[v_seq_end:j_seq_end]

    # Pad front if V germline doesn't start at position 0
    if v_germ_start > 0:
        seq_to_gap = '.' * v_germ_start + seq_to_gap

    # Walk through the V reference and insert gaps where the reference has them
    seq_iter = iter(seq_to_gap)
    aligned = []
    started = False
    for ref_base in v_ref_trimmed:
        if ref_base != '.':
            started = True
            aligned.append(next(seq_iter, '.'))
        else:
            aligned.append('.' if started else next(seq_iter, '.'))

    return ''.join(aligned) + seq_remain


# ---------------------------------------------------------------------------
# NP region extraction
# ---------------------------------------------------------------------------

def compute_np_regions(
    seq: str,
    v_seq_end: int,
    j_seq_start: int,
    d_seq_start: Optional[int],
    d_seq_end: Optional[int],
    chain: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Extract NP1 and NP2 nucleotide sequences.

    Heavy chain: NP1 = V_end..D_start, NP2 = D_end..J_start
    Light chain: NP1 = V_end..J_start, NP2 = None
    """
    if chain == 'heavy':
        np1 = seq[v_seq_end:d_seq_start] if d_seq_start is not None else ''
        np2 = seq[d_seq_end:j_seq_start] if d_seq_end is not None else ''
        return np1, np2
    else:
        np1 = seq[v_seq_end:j_seq_start]
        return np1, None


# ---------------------------------------------------------------------------
# Germline alignment
# ---------------------------------------------------------------------------

def build_germline_alignment(
    seq: str,
    refs: ReferenceData,
    v_call: str,
    j_call: str,
    d_call: Optional[str],
    v_germ_end: int,
    j_germ_start: int,
    j_germ_end: int,
    d_germ_start: Optional[int],
    d_germ_end: Optional[int],
    np1: Optional[str],
    np2: Optional[str],
    v_seq_end: int,
    j_seq_start: int,
    chain: str,
) -> Optional[str]:
    """Build the germline alignment string.

    Concatenates: V_germline + middle_region + J_germline.
    Middle region depends on chain type and D call.
    """
    # V germline
    v_ref = refs.v_gapped.get(v_call.split(',')[0], '')
    v_part = _get_reference_segment(v_ref, v_germ_end)

    # J germline
    j_ref_full = refs.j_ungapped.get(j_call.split(',')[0], '')
    j_part = j_ref_full[j_germ_start:j_germ_end + _count_gaps(j_ref_full)] if j_ref_full else ''

    if chain == 'heavy':
        first_d_call = d_call.split(',')[0] if d_call else ''
        if SHORT_D_SENTINEL in first_d_call:
            # Short-D: use the raw sequence between V and J
            d_region = seq[v_seq_end:j_seq_start]
        else:
            # Normal D: NP1 + D_germline + NP2
            d_ref_full = refs.d_ungapped.get(first_d_call, '')
            d_part = d_ref_full[d_germ_start:d_germ_end + _count_gaps(d_ref_full)] if d_ref_full else ''
            d_region = (np1 or '') + d_part + (np2 or '')
        return v_part + d_region + j_part
    else:
        return v_part + (np1 or '') + j_part


# ---------------------------------------------------------------------------
# Alignment position mapping
# ---------------------------------------------------------------------------

def compute_alignment_positions(
    v_ref_gapped: str,
    v_germ_end: int,
    v_seq_start: int,
    d_seq_start: Optional[int],
    d_seq_end: Optional[int],
    j_seq_start: int,
    j_seq_end: int,
    chain: str,
) -> Dict[str, Optional[int]]:
    """Map raw sequence positions to IMGT-alignment coordinates.

    Returns a dict with keys: v_alignment_start, v_alignment_end,
    d_alignment_start, d_alignment_end, j_alignment_start, j_alignment_end,
    v_germline_start, v_germline_end.
    """
    gap = _count_gaps(v_ref_gapped) if v_ref_gapped else 0
    v_end = v_germ_end + gap

    d_start_al = None
    d_end_al = None
    if chain == 'heavy' and d_seq_start is not None:
        d_start_al = d_seq_start - v_seq_start + gap
        d_end_al = d_seq_end - v_seq_start + gap

    j_start_al = j_seq_start - v_seq_start + gap
    j_end_al = j_seq_end - v_seq_start + gap

    return {
        'v_alignment_start': 0,
        'v_alignment_end': v_end,
        'v_germline_start': 0,
        'v_germline_end': v_end,
        'd_alignment_start': d_start_al,
        'd_alignment_end': d_end_al,
        'j_alignment_start': j_start_al,
        'j_alignment_end': j_end_al,
    }


# ---------------------------------------------------------------------------
# Segment alignment extraction
# ---------------------------------------------------------------------------

def extract_segment_alignments(
    seq_alignment: str,
    germ_alignment: str,
    seq_alignment_aa: str,
    germ_alignment_aa: str,
    alignment_positions: Dict[str, Optional[int]],
    chain: str,
) -> dict:
    """Extract per-segment alignment substrings and their AA translations.

    Returns a dict with keys like v_sequence_alignment, v_germline_alignment,
    v_sequence_alignment_aa, v_germline_alignment_aa, and similarly for d/j.
    """
    result = {}
    segments = ['v', 'j'] + (['d'] if chain == 'heavy' else [])

    for seg in segments:
        s = alignment_positions.get(f'{seg}_alignment_start')
        e = alignment_positions.get(f'{seg}_alignment_end')

        if s is not None and e is not None:
            result[f'{seg}_sequence_alignment'] = seq_alignment[s:e]
            result[f'{seg}_germline_alignment'] = germ_alignment[s:e]
            result[f'{seg}_sequence_alignment_aa'] = seq_alignment_aa[s // 3:e // 3]
            result[f'{seg}_germline_alignment_aa'] = germ_alignment_aa[s // 3:e // 3]
        else:
            result[f'{seg}_sequence_alignment'] = None
            result[f'{seg}_germline_alignment'] = None
            result[f'{seg}_sequence_alignment_aa'] = None
            result[f'{seg}_germline_alignment_aa'] = None

    return result


# ---------------------------------------------------------------------------
# Translation wrapper
# ---------------------------------------------------------------------------

def translate_alignment(alignment: Optional[str]) -> Optional[str]:
    """Translate a nucleotide alignment to amino acids using GenAIRR."""
    if alignment is None:
        return None
    return translate(alignment)
