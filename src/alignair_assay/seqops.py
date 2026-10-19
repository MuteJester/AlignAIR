"""Nucleotide helpers: reverse complement and standard-code translation."""
from __future__ import annotations

from itertools import product

_COMPLEMENT = str.maketrans("ACGTRYKMSWBDHVNacgtrykmswbdhvn", "TGCAYRMKSWVHDBNtgcayrmkswvhdbn")

_BASES = "TCAG"
_AMINO = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
_CODONS = {"".join(c): aa for c, aa in zip(product(_BASES, repeat=3), _AMINO)}


def revcomp(s: str) -> str:
    """Reverse complement (IUPAC-aware, case preserved)."""
    return s.translate(_COMPLEMENT)[::-1]


def translate(s: str) -> str:
    """Translate full codons with the standard code; non-ACGT codons give ``X``, a partial tail is dropped."""
    s = s.upper()
    return "".join(_CODONS.get(s[i:i + 3], "X") for i in range(0, len(s) - 2, 3))
