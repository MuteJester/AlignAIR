"""Constants for AIRR rearrangement formatting."""
from __future__ import annotations

# IMGT-numbered region boundaries (0-based on the IMGT-gapped alignment). None = per-sequence.
IMGT_REGIONS = {
    "fwr1": (0, 78),
    "cdr1": (78, 114),
    "fwr2": (114, 165),
    "cdr2": (165, 195),
    "fwr3": (195, 312),
    "cdr3": (312, None),
    "fwr4": (None, None),
    "junction": (309, None),
}

SHORT_D_SENTINEL = "Short-D"

AIRR_BOOLEAN_COLUMNS = ("stop_codon", "vj_in_frame", "productive")
