"""FWR/CDR region extraction and junction/CDR3 computation."""
from __future__ import annotations

from typing import Dict, Optional

from AlignAIR.Pipeline.AIRR.constants import IMGT_REGIONS


def extract_regions(
    seq_alignment: Optional[str],
    seq_alignment_aa: Optional[str],
) -> dict:
    """Extract FWR1-3 and CDR1-2 from the IMGT-gapped alignment.

    Uses fixed IMGT positions. CDR3, FWR4, and junction are handled
    separately by compute_junction() because they require J-anchor info.

    Returns a dict with keys: fwr1, fwr1_aa, fwr1_start, fwr1_end,
    cdr1, cdr1_aa, cdr1_start, cdr1_end, ... fwr3, fwr3_aa, fwr3_start, fwr3_end.
    """
    result = {}
    if seq_alignment is None:
        for region in ('fwr1', 'cdr1', 'fwr2', 'cdr2', 'fwr3'):
            result[region] = None
            result[f'{region}_aa'] = None
            result[f'{region}_start'] = None
            result[f'{region}_end'] = None
        return result

    for region in ('fwr1', 'cdr1', 'fwr2', 'cdr2', 'fwr3'):
        start, end = IMGT_REGIONS[region]
        result[region] = seq_alignment[start:end]
        result[f'{region}_start'] = start
        result[f'{region}_end'] = end

        aa_start = start // 3 if start is not None else None
        aa_end = end // 3 if end is not None else None
        if seq_alignment_aa is not None:
            result[f'{region}_aa'] = seq_alignment_aa[aa_start:aa_end]
        else:
            result[f'{region}_aa'] = None

    return result


def compute_junction(
    seq_alignment: Optional[str],
    seq_alignment_aa: Optional[str],
    j_call: str,
    j_anchor_dict: Dict[str, int],
    j_seq_start: int,
    v_seq_start: int,
    j_germ_start: int,
    j_alignment_end: Optional[int],
) -> dict:
    """Compute junction, CDR3, and FWR4 from the aligned sequence.

    The junction end is determined by the conserved J anchor position:
        junc_end = j_seq_start - v_seq_start + gap_count + j_anchor - j_germ_start + 3

    Returns a dict with junction, junction_aa, junction_length,
    junction_aa_length, cdr3, cdr3_aa, cdr3_start, cdr3_end,
    fwr4, fwr4_aa, fwr4_start, fwr4_end.
    """
    null_result = {
        'junction': None, 'junction_aa': None,
        'junction_length': None, 'junction_aa_length': None,
        'cdr3': None, 'cdr3_aa': None, 'cdr3_start': None, 'cdr3_end': None,
        'fwr4': None, 'fwr4_aa': None, 'fwr4_start': None, 'fwr4_end': None,
    }

    if seq_alignment is None or seq_alignment_aa is None:
        return null_result

    first_j = j_call.split(',')[0]
    j_anchor = j_anchor_dict.get(first_j, 0)
    offset = seq_alignment.count('.')

    junction_start = IMGT_REGIONS['junction'][0]
    junc_end = j_seq_start - v_seq_start + offset + j_anchor - j_germ_start + 3

    junction_nt = seq_alignment[junction_start:junc_end]
    junction_aa = seq_alignment_aa[junction_start // 3:junc_end // 3]
    cdr3_aa = junction_aa[1:-1] if len(junction_aa) > 2 else ''

    cdr3_start = junction_start + 3
    cdr3_end = junc_end - 3
    cdr3_nt = seq_alignment[cdr3_start:cdr3_end]

    fwr4_start = cdr3_end + 1
    fwr4_end = j_alignment_end
    fwr4_nt = seq_alignment[fwr4_start:fwr4_end] if fwr4_end is not None else None
    fwr4_aa = (
        seq_alignment_aa[fwr4_start // 3:fwr4_end // 3]
        if fwr4_end is not None and seq_alignment_aa is not None
        else None
    )

    return {
        'junction': junction_nt,
        'junction_aa': junction_aa,
        'junction_length': len(junction_nt),
        'junction_aa_length': len(junction_aa),
        'cdr3': cdr3_nt,
        'cdr3_aa': cdr3_aa,
        'cdr3_start': cdr3_start,
        'cdr3_end': cdr3_end,
        'fwr4': fwr4_nt,
        'fwr4_aa': fwr4_aa,
        'fwr4_start': fwr4_start,
        'fwr4_end': fwr4_end,
    }
