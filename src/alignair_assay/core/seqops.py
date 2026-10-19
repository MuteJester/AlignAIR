"""Nucleotide helpers: reverse complement and standard-code translation."""
from __future__ import annotations

from itertools import product

_COMPLEMENT = str.maketrans("ACGTRYKMSWBDHVNacgtrykmswbdhvn", "TGCAYRMKSWVHDBNtgcayrmkswvhdbn")

_BASES = "TCAG"
_AMINO = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
_CODONS = {"".join(c): aa for c, aa in zip(product(_BASES, repeat=3), _AMINO)}

_IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T", "R": "AG", "Y": "CT", "K": "GT", "M": "AC", "S": "CG",
          "W": "AT", "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT"}


def _shared_amino_acid(codon: tuple[str, ...]) -> str:
    """The amino acid every expansion of an IUPAC codon encodes, else ``X``."""
    amino = {_CODONS["".join(c)] for c in product(*(_IUPAC[b] for b in codon))}
    return amino.pop() if len(amino) == 1 else "X"


_TRANSLATION = {"".join(c): _shared_amino_acid(c) for c in product(_IUPAC, repeat=3)}


def revcomp(s: str) -> str:
    """Reverse complement (IUPAC-aware, case preserved)."""
    return s.translate(_COMPLEMENT)[::-1]


def translate(s: str) -> str:
    """Translate full codons with the standard code; a partial tail is dropped.

    A codon with IUPAC ambiguity codes gives the amino acid shared by all its expansions (``CGN`` -> ``R``), else
    ``X`` (``TTN``, ``NNN``); a codon with any other character gives ``X``.
    """
    s = s.upper()
    return "".join(_TRANSLATION.get(s[i:i + 3], "X") for i in range(0, len(s) - 2, 3))
