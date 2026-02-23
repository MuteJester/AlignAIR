"""Constants for AIRR rearrangement formatting."""
from __future__ import annotations

# IMGT-numbered region boundaries (0-based positions on the IMGT-gapped alignment).
# None means "determined per-sequence" (e.g. cdr3 end depends on the junction).
IMGT_REGIONS = {
    'fwr1': (0, 78),
    'cdr1': (78, 114),
    'fwr2': (114, 165),
    'cdr2': (165, 195),
    'fwr3': (195, 312),
    'cdr3': (312, None),   # end is per-sequence
    'fwr4': (None, None),  # start/end are per-sequence
    'junction': (309, None),
}

# Magic sentinel for D genes that are too short to align.
SHORT_D_SENTINEL = "Short-D"

# AIRR-required column order (standard + AlignAIR extras).
AIRR_REQUIRED_COLUMNS = [
    'sequence_id', 'sequence', 'locus', 'stop_codon', 'vj_in_frame', 'productive',
    'v_call', 'd_call', 'j_call', 'sequence_alignment', 'germline_alignment',
    'sequence_alignment_aa', 'germline_alignment_aa',
    'v_sequence_alignment', 'v_sequence_alignment_aa', 'v_germline_alignment',
    'v_germline_alignment_aa', 'd_sequence_alignment', 'd_sequence_alignment_aa',
    'd_germline_alignment', 'd_germline_alignment_aa', 'j_sequence_alignment',
    'j_sequence_alignment_aa', 'j_germline_alignment', 'j_germline_alignment_aa',
    'fwr1', 'fwr1_aa', 'cdr1', 'cdr1_aa', 'fwr2', 'fwr2_aa', 'cdr2', 'cdr2_aa',
    'fwr3', 'fwr3_aa', 'fwr4', 'fwr4_aa', 'cdr3', 'cdr3_aa', 'junction', 'junction_length',
    'junction_aa', 'junction_aa_length', 'v_sequence_start', 'v_sequence_end',
    'v_germline_start', 'v_germline_end', 'v_alignment_start', 'v_alignment_end',
    'd_sequence_start', 'd_sequence_end', 'd_germline_start', 'd_germline_end',
    'd_alignment_start', 'd_alignment_end', 'j_sequence_start', 'j_sequence_end',
    'j_germline_start', 'j_germline_end', 'j_alignment_start', 'j_alignment_end',
    'fwr1_start', 'fwr1_end', 'cdr1_start', 'cdr1_end', 'fwr2_start', 'fwr2_end',
    'cdr2_start', 'cdr2_end', 'fwr3_start', 'fwr3_end', 'fwr4_start', 'fwr4_end',
    'cdr3_start', 'cdr3_end', 'np1', 'np1_length', 'np2', 'np2_length',
]

AIRR_EXTRA_COLUMNS = [
    'v_likelihoods', 'd_likelihoods', 'j_likelihoods', 'mutation_rate', 'ar_indels',
]

# Columns whose boolean values are serialized as 'T'/'F' in AIRR TSV output.
AIRR_BOOLEAN_COLUMNS = ['stop_codon', 'vj_in_frame', 'productive']
