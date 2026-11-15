"""Constants for AIRR rearrangement formatting."""
from __future__ import annotations

# The IMGT-numbered FWR and CDR columns of the V (0-based, end-exclusive, on the IMGT-gapped alignment); the CDR3 and
# FWR4 are the read's own, from its junction (``regions.read_junction_fields``).
IMGT_REGIONS = {
    "fwr1": (0, 78),
    "cdr1": (78, 114),
    "fwr2": (114, 165),
    "cdr2": (165, 195),
    "fwr3": (195, 312),
}

SHORT_D_SENTINEL = "Short-D"
