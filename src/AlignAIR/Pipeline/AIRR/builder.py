"""Orchestrator — builds complete AIRR DataFrames with per-sequence error handling."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from AlignAIR.Pipeline.AIRR.constants import (
    AIRR_BOOLEAN_COLUMNS,
    AIRR_EXTRA_COLUMNS,
    AIRR_REQUIRED_COLUMNS,
    SHORT_D_SENTINEL,
)
from AlignAIR.Pipeline.AIRR.references import ReferenceData, build_reference_maps
from AlignAIR.Pipeline.AIRR.alignment import (
    build_germline_alignment,
    build_sequence_alignment,
    compute_alignment_positions,
    compute_np_regions,
    extract_segment_alignments,
    translate_alignment,
)
from AlignAIR.Pipeline.AIRR.regions import compute_junction, extract_regions
from AlignAIR.Pipeline.AIRR.quality import compute_v_identity, has_stop_codon, is_vj_in_frame

logger = logging.getLogger("AlignAIR.Pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_list(x, n: int) -> list:
    """Coerce value to a plain Python list of length n."""
    if isinstance(x, np.ndarray):
        return x.reshape(-1).tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x] * n


def _safe_int(val) -> int:
    """Convert an indel count value to an integer, defaulting to 0."""
    try:
        if val is None:
            return 0
        if isinstance(val, (list, tuple, np.ndarray)):
            return int(val[0]) if len(val) > 0 else 0
        s = str(val)
        if s.lower() == 'nan' or s == '':
            return 0
        return int(float(val))
    except Exception:
        return 0


def _determine_locus(
    chain: str,
    dataconfig,
    processed_predictions: dict,
    n: int,
    index: int,
) -> str:
    """Determine the AIRR locus string for a single sequence."""
    if chain == 'heavy':
        return 'IGH'

    # Light chain: check if type_ prediction is present (multi-chain model)
    if 'type_' in processed_predictions:
        try:
            t = np.array(processed_predictions['type_']).astype(int).squeeze()
            if t.ndim == 0:
                return 'IGK' if int(t) == 1 else 'IGL'
            return 'IGK' if int(t[index]) == 1 else 'IGL'
        except Exception:
            pass
    return 'IGL'


def _format_likelihoods(arr) -> str:
    """Convert a likelihood array to a semicolon-delimited string."""
    if arr is None:
        return ''
    return ';'.join(f'{x:.6f}' for x in arr)


# ---------------------------------------------------------------------------
# Per-sequence AIRR record builder
# ---------------------------------------------------------------------------

def _build_single_record(
    i: int,
    seq: str,
    refs: ReferenceData,
    allele_calls: dict,
    germline: dict,
    likelihoods: dict,
    preds: dict,
    dataconfig,
    skip_processing: bool,
) -> dict:
    """Build a complete AIRR record dict for a single sequence.

    Returns a flat dict with all AIRR fields. If skip_processing is True,
    alignment-derived fields are set to None.
    """
    chain = refs.chain

    # Basic calls
    v_call = ','.join(allele_calls['v'][i])
    j_call = ','.join(allele_calls['j'][i])
    d_call = ','.join(allele_calls['d'][i]) if refs.has_d else ''

    # Positions
    v_g = germline['v'][i]
    j_g = germline['j'][i]
    v_seq_start = v_g['start_in_seq']
    v_seq_end = v_g['end_in_seq']
    v_germ_start = max(0, v_g['start_in_ref'])
    v_germ_end = v_g['end_in_ref']
    j_seq_start = j_g['start_in_seq']
    j_seq_end = j_g['end_in_seq']
    j_germ_start = max(0, j_g['start_in_ref'])
    j_germ_end = j_g['end_in_ref']

    d_seq_start = None
    d_seq_end = None
    d_germ_start = None
    d_germ_end = None
    if refs.has_d:
        d_g = germline['d'][i]
        d_seq_start = d_g['start_in_seq']
        d_seq_end = d_g['end_in_seq']
        d_germ_start = abs(d_g['start_in_ref'])
        d_germ_end = d_g['end_in_ref']

    # Scalars
    productive = preds['productive'][i] if i < len(preds['productive']) else False
    mutation_rate = preds['mutation_rate'][i] if i < len(preds['mutation_rate']) else 0.0
    raw_indel = preds.get('indel_count', [0])[i] if i < len(preds.get('indel_count', [])) else 0
    indel_count = _safe_int(raw_indel)

    # Likelihoods
    v_lk = _format_likelihoods(likelihoods.get('v', [None] * (i + 1))[i])
    j_lk = _format_likelihoods(likelihoods.get('j', [None] * (i + 1))[i])
    d_lk = _format_likelihoods(likelihoods.get('d', [None] * (i + 1))[i]) if refs.has_d else None

    # Locus
    locus = _determine_locus(chain, dataconfig, preds, 0, i)

    # Build the base record
    record = {
        'sequence': seq,
        'v_call': v_call,
        'd_call': d_call,
        'j_call': j_call,
        'v_sequence_start': v_seq_start,
        'v_sequence_end': v_seq_end,
        'v_germline_start': v_germ_start,
        'v_germline_end': v_germ_end,
        'j_sequence_start': j_seq_start,
        'j_sequence_end': j_seq_end,
        'j_germline_start': j_germ_start,
        'j_germline_end': j_germ_end,
        'd_sequence_start': d_seq_start,
        'd_sequence_end': d_seq_end,
        'd_germline_start': d_germ_start,
        'd_germline_end': d_germ_end,
        'locus': locus,
        'productive': productive,
        'mutation_rate': mutation_rate,
        'ar_indels': indel_count,
        'v_likelihoods': v_lk,
        'j_likelihoods': j_lk,
        'd_likelihoods': d_lk,
    }

    # If skip_processing, set all alignment-derived fields to None
    if skip_processing:
        for key in (
            'sequence_alignment', 'germline_alignment',
            'sequence_alignment_aa', 'germline_alignment_aa',
            'np1', 'np2', 'np1_length', 'np2_length',
        ):
            record[key] = None
        # Null out alignment positions
        for prefix in ('v', 'd', 'j'):
            record[f'{prefix}_alignment_start'] = None
            record[f'{prefix}_alignment_end'] = None
        # Null out segment alignments
        for seg in ('v', 'd', 'j'):
            for suffix in ('_sequence_alignment', '_germline_alignment',
                           '_sequence_alignment_aa', '_germline_alignment_aa'):
                record[seg + suffix] = None
        # Null out regions
        for region in ('fwr1', 'cdr1', 'fwr2', 'cdr2', 'fwr3'):
            record[region] = None
            record[f'{region}_aa'] = None
            record[f'{region}_start'] = None
            record[f'{region}_end'] = None
        # Null out junction/CDR3/FWR4
        for key in ('junction', 'junction_aa', 'junction_length', 'junction_aa_length',
                     'cdr3', 'cdr3_aa', 'cdr3_start', 'cdr3_end',
                     'fwr4', 'fwr4_aa', 'fwr4_start', 'fwr4_end'):
            record[key] = None
        # Quality flags
        record['stop_codon'] = False
        record['vj_in_frame'] = False
        return record

    # ---- Alignment-derived fields ----

    # 1. Sequence alignment
    v_ref_gapped = refs.v_gapped.get(v_call.split(',')[0], '')
    seq_alignment = build_sequence_alignment(
        seq, v_ref_gapped, v_seq_start, v_seq_end,
        v_germ_start, v_germ_end, j_seq_end,
    )
    record['sequence_alignment'] = seq_alignment

    # 2. NP regions
    np1, np2 = compute_np_regions(seq, v_seq_end, j_seq_start, d_seq_start, d_seq_end, chain)
    record['np1'] = np1
    record['np2'] = np2
    record['np1_length'] = len(np1) if np1 is not None else None
    record['np2_length'] = len(np2) if np2 is not None else None

    # 3. Germline alignment
    germ_alignment = build_germline_alignment(
        seq, refs, v_call, j_call, d_call,
        v_germ_end, j_germ_start, j_germ_end,
        d_germ_start, d_germ_end,
        np1, np2, v_seq_end, j_seq_start, chain,
    )
    record['germline_alignment'] = germ_alignment

    # 4. Translations
    seq_alignment_aa = translate_alignment(seq_alignment)
    germ_alignment_aa = translate_alignment(germ_alignment)
    record['sequence_alignment_aa'] = seq_alignment_aa
    record['germline_alignment_aa'] = germ_alignment_aa

    # 5. Alignment positions
    al_pos = compute_alignment_positions(
        v_ref_gapped, v_germ_end, v_seq_start,
        d_seq_start, d_seq_end, j_seq_start, j_seq_end, chain,
    )
    record.update(al_pos)

    # 6. Segment alignments
    if seq_alignment and germ_alignment and seq_alignment_aa and germ_alignment_aa:
        seg_als = extract_segment_alignments(
            seq_alignment, germ_alignment,
            seq_alignment_aa, germ_alignment_aa,
            al_pos, chain,
        )
        record.update(seg_als)
    else:
        for seg in ('v', 'd', 'j'):
            for suffix in ('_sequence_alignment', '_germline_alignment',
                           '_sequence_alignment_aa', '_germline_alignment_aa'):
                record[seg + suffix] = None

    # 7. Regions (FWR1-3, CDR1-2)
    regions = extract_regions(seq_alignment, seq_alignment_aa)
    record.update(regions)

    # 8. Junction, CDR3, FWR4
    junc = compute_junction(
        seq_alignment, seq_alignment_aa, j_call, refs.j_anchors,
        j_seq_start, v_seq_start, j_germ_start,
        al_pos.get('j_alignment_end'),
    )
    record.update(junc)

    # 9. Quality flags
    record['stop_codon'] = has_stop_codon(seq_alignment_aa)
    record['vj_in_frame'] = is_vj_in_frame(
        record.get('cdr3_start'), record.get('cdr3_end'),
        al_pos.get('v_alignment_start'),
    )

    return record


# ---------------------------------------------------------------------------
# Finalization (1-based indexing, T/F booleans, column ordering)
# ---------------------------------------------------------------------------

def _finalize_airr_dict(airr_dict: Dict[str, list], n: int) -> Dict[str, list]:
    """Apply AIRR formatting conventions to the columnar dict.

    - Fill missing columns with None
    - Convert _start columns to 1-based indexing
    - Convert boolean columns to 'T'/'F'
    """
    all_cols = AIRR_REQUIRED_COLUMNS + AIRR_EXTRA_COLUMNS
    for col in all_cols:
        if col not in airr_dict:
            airr_dict[col] = [None] * n

    # 1-based indexing for all _start columns
    for col in list(airr_dict.keys()):
        if col.endswith('_start'):
            airr_dict[col] = [
                (x + 1 if x is not None else None) for x in airr_dict[col]
            ]

    # Boolean → 'T'/'F'
    for col in AIRR_BOOLEAN_COLUMNS:
        if col in airr_dict:
            airr_dict[col] = [
                'T' if v else 'F' if v is not None else None
                for v in airr_dict[col]
            ]

    # Reorder to standard column order, keeping any extras at the end
    ordered = {}
    for col in all_cols:
        if col in airr_dict:
            ordered[col] = airr_dict[col]
    # Append any columns not in the standard list
    for col in airr_dict:
        if col not in ordered:
            ordered[col] = airr_dict[col]
    return ordered


# ---------------------------------------------------------------------------
# Public API: build_airr_dataframe
# ---------------------------------------------------------------------------

def build_airr_dataframe(
    sequences: List[str],
    allele_calls: dict,
    germline_alignments: dict,
    likelihoods: dict,
    processed_predictions: dict,
    dataconfig,
    sequence_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build a complete AIRR-formatted DataFrame.

    Args:
        sequences: Input nucleotide sequences.
        allele_calls: {'v': [[allele_names], ...], 'd': [...], 'j': [...]}.
        germline_alignments: {'v': [pos_dicts], 'd': [pos_dicts], 'j': [pos_dicts]}.
        likelihoods: {'v': [np.ndarray], 'd': [...], 'j': [...]}.
        processed_predictions: Dict with 'mutation_rate', 'indel_count', 'productive',
                                optionally 'type_'.
        dataconfig: GenAIRR MultiDataConfigContainer (has .packaged_config()).
        sequence_ids: Optional sequence IDs. If None, uses 0-based indices.

    Returns:
        pd.DataFrame in AIRR standard format.
    """
    n = len(sequences)
    refs = build_reference_maps(dataconfig)

    # Flatten prediction arrays
    preds_flat = {
        'productive': _to_list(processed_predictions.get('productive', [False] * n), n),
        'mutation_rate': _to_list(processed_predictions.get('mutation_rate', [0.0] * n), n),
        'indel_count': _to_list(processed_predictions.get('indel_count', [0] * n), n),
    }
    if 'type_' in processed_predictions:
        preds_flat['type_'] = processed_predictions['type_']

    # Determine skip_processing per sequence
    indel_ints = [_safe_int(x) for x in preds_flat['indel_count']]
    skip_flags = [
        (p is False and ic > 1)
        for p, ic in zip(preds_flat['productive'], indel_ints)
    ]

    # Build records
    columns: Dict[str, list] = {}
    failed_indices = []

    for i in range(n):
        try:
            record = _build_single_record(
                i, sequences[i], refs, allele_calls, germline_alignments,
                likelihoods, preds_flat, dataconfig, skip_flags[i],
            )
        except Exception as e:
            logger.warning("AIRR record failed for sequence %d: %s", i, e)
            failed_indices.append(i)
            record = {'sequence': sequences[i]}

        # Accumulate into columnar dict
        for key, value in record.items():
            if key not in columns:
                # Backfill previous rows with None
                columns[key] = [None] * i
            columns[key].append(value)

        # Forward-fill any columns this record didn't produce
        for key in columns:
            if len(columns[key]) <= i:
                columns[key].append(None)

    # Sequence IDs
    if sequence_ids is not None:
        columns['sequence_id'] = list(sequence_ids)
    else:
        columns['sequence_id'] = list(range(n))

    if failed_indices:
        logger.warning("%d sequences failed AIRR formatting: %s",
                       len(failed_indices), failed_indices[:10])

    # Finalize
    columns = _finalize_airr_dict(columns, n)

    return pd.DataFrame(columns)


# ---------------------------------------------------------------------------
# Public API: build_csv_enrichment
# ---------------------------------------------------------------------------

def build_csv_enrichment(
    sequences: List[str],
    allele_calls: dict,
    germline_alignments: dict,
    likelihoods: dict,
    processed_predictions: dict,
    dataconfig,
) -> dict:
    """Compute enrichment fields for CSV output.

    Returns a dict of lists (one per sequence) with keys:
    sequence_length, locus, junction, junction_aa, junction_length,
    np1_length, np2_length, v_sequence, d_sequence, j_sequence,
    v_identity, stop_codon, vj_in_frame.
    """
    n = len(sequences)
    refs = build_reference_maps(dataconfig)

    preds_flat = {
        'productive': _to_list(processed_predictions.get('productive', [False] * n), n),
        'mutation_rate': _to_list(processed_predictions.get('mutation_rate', [0.0] * n), n),
        'indel_count': _to_list(processed_predictions.get('indel_count', [0] * n), n),
    }
    if 'type_' in processed_predictions:
        preds_flat['type_'] = processed_predictions['type_']

    indel_ints = [_safe_int(x) for x in preds_flat['indel_count']]
    skip_flags = [
        (p is False and ic > 1)
        for p, ic in zip(preds_flat['productive'], indel_ints)
    ]

    result = {
        'sequence_length': [], 'locus': [],
        'junction': [], 'junction_aa': [], 'junction_length': [],
        'np1_length': [], 'np2_length': [],
        'v_sequence': [], 'd_sequence': [], 'j_sequence': [],
        'v_identity': [], 'stop_codon': [], 'vj_in_frame': [],
    }

    for i in range(n):
        seq = sequences[i]
        result['sequence_length'].append(len(seq))
        result['locus'].append(_determine_locus(refs.chain, dataconfig, preds_flat, n, i))

        v_g = germline_alignments['v'][i]
        j_g = germline_alignments['j'][i]
        v_seq_start = v_g['start_in_seq']
        v_seq_end = v_g['end_in_seq']
        v_germ_start = max(0, v_g['start_in_ref'])
        v_germ_end = v_g['end_in_ref']
        j_seq_start = j_g['start_in_seq']
        j_seq_end = j_g['end_in_seq']
        j_germ_start = max(0, j_g['start_in_ref'])

        d_seq_start = None
        d_seq_end = None
        if refs.has_d:
            d_g = germline_alignments['d'][i]
            d_seq_start = d_g['start_in_seq']
            d_seq_end = d_g['end_in_seq']

        # Segment sequences
        result['v_sequence'].append(seq[v_seq_start:v_seq_end])
        result['j_sequence'].append(seq[j_seq_start:j_seq_end])
        result['d_sequence'].append(seq[d_seq_start:d_seq_end] if d_seq_start is not None else None)

        # NP regions
        np1, np2 = compute_np_regions(seq, v_seq_end, j_seq_start, d_seq_start, d_seq_end, refs.chain)
        result['np1_length'].append(len(np1) if np1 is not None else None)
        result['np2_length'].append(len(np2) if np2 is not None else None)

        if skip_flags[i]:
            result['junction'].append(None)
            result['junction_aa'].append(None)
            result['junction_length'].append(None)
            result['v_identity'].append(None)
            result['stop_codon'].append(None)
            result['vj_in_frame'].append(None)
            continue

        # Build alignment for junction + quality computation
        try:
            v_call = ','.join(allele_calls['v'][i])
            j_call = ','.join(allele_calls['j'][i])
            v_ref_gapped = refs.v_gapped.get(v_call.split(',')[0], '')

            seq_alignment = build_sequence_alignment(
                seq, v_ref_gapped, v_seq_start, v_seq_end,
                v_germ_start, v_germ_end, j_seq_end,
            )

            seq_alignment_aa = translate_alignment(seq_alignment)

            # Alignment positions for junction calc
            al_pos = compute_alignment_positions(
                v_ref_gapped, v_germ_end, v_seq_start,
                d_seq_start, d_seq_end, j_seq_start, j_seq_end, refs.chain,
            )

            # Junction
            junc = compute_junction(
                seq_alignment, seq_alignment_aa, j_call, refs.j_anchors,
                j_seq_start, v_seq_start, j_germ_start,
                al_pos.get('j_alignment_end'),
            )
            result['junction'].append(junc['junction'])
            result['junction_aa'].append(junc['junction_aa'])
            result['junction_length'].append(junc['junction_length'])

            # V identity — build germline alignment once
            d_call_str = ','.join(allele_calls['d'][i]) if refs.has_d else ''
            d_germ_start_i = abs(germline_alignments['d'][i]['start_in_ref']) if refs.has_d else None
            d_germ_end_i = germline_alignments['d'][i]['end_in_ref'] if refs.has_d else None
            germ_alignment = build_germline_alignment(
                seq, refs, v_call, j_call, d_call_str,
                v_germ_end, j_germ_start, j_g['end_in_ref'],
                d_germ_start_i, d_germ_end_i,
                np1, np2, v_seq_end, j_seq_start, refs.chain,
            )
            germ_alignment_aa = translate_alignment(germ_alignment)

            seg_als = extract_segment_alignments(
                seq_alignment, germ_alignment,
                seq_alignment_aa, germ_alignment_aa,
                al_pos, refs.chain,
            )
            result['v_identity'].append(
                compute_v_identity(
                    seg_als.get('v_sequence_alignment'),
                    seg_als.get('v_germline_alignment'),
                )
            )

            # Quality flags
            result['stop_codon'].append(has_stop_codon(seq_alignment_aa))
            result['vj_in_frame'].append(is_vj_in_frame(
                junc.get('cdr3_start'), junc.get('cdr3_end'),
                al_pos.get('v_alignment_start'),
            ))

        except Exception as e:
            logger.warning("CSV enrichment failed for sequence %d: %s", i, e)
            result['junction'].append(None)
            result['junction_aa'].append(None)
            result['junction_length'].append(None)
            result['v_identity'].append(None)
            result['stop_codon'].append(None)
            result['vj_in_frame'].append(None)

    return result
