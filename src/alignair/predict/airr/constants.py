"""Constants for AIRR rearrangement formatting."""
from __future__ import annotations

# The IMGT-numbered FWR and CDR columns of the V (0-based, end-exclusive, on the IMGT-gapped alignment), which cut the
# region's bases from the alignment; a region's start and end are read positions (``regions.extract_regions``). The
# CDR3 and FWR4 (IMGT's FR4, from the J Trp or Phe codon) are the read's own, from its junction
# (``regions.read_junction_fields``).
IMGT_REGIONS = {
    "fwr1": (0, 78),
    "cdr1": (78, 114),
    "fwr2": (114, 165),
    "cdr2": (165, 195),
    "fwr3": (195, 312),
}
