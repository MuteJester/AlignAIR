"""Nucleotide to amino-acid translation (the AIRR row's junction_aa and CDR3/FWR translations)."""
from __future__ import annotations

# Standard genetic code (NCBI transl_table=1). Codons enumerated with base order T,C,A,G
# so they align with the canonical amino-acid string below.
_BASES = "TCAG"
_AAS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE: dict[str, str] = {
    x + y + z: aa
    for (x, y, z), aa in zip(
        ((a, b, c) for a in _BASES for b in _BASES for c in _BASES), _AAS
    )
}


def translate_nt(nt: str, frame: int = 0, unknown: str = "X", stop: str = "*") -> str:
    """Translate ``nt`` from ``frame``; trailing partial codon dropped; non-ACGT codon -> ``unknown``."""
    s = nt[frame:].upper().replace("U", "T")
    out = []
    for i in range(0, len(s) - 2, 3):
        aa = CODON_TABLE.get(s[i:i + 3], unknown)
        out.append(stop if aa == "*" else aa)
    return "".join(out)
