"""Amino-acid translation + codon-provenance projection.

Coordinate projection is NOT ``//3``: GenAIRR translates the assembled read in a
junction-derived reading frame and drops incomplete boundary codons. A residue is
assigned to a nucleotide segment only when ALL THREE of its codon's nucleotides lie
inside that segment; codons that straddle a segment boundary (or land in an N/P gap)
are masked (provenance ``None``).
"""
from __future__ import annotations

from dataclasses import dataclass

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


def frame_offset_of(junction_start: int | None) -> int | None:
    """Reading-frame offset of the assembled read = ``junction_start % 3`` (V-anchor frame)."""
    return None if junction_start is None else junction_start % 3


@dataclass(frozen=True)
class AAProjection:
    n_residues: int
    provenance: tuple                     # per residue: "V"/"D"/"J"/None (None = masked)
    aa_bounds: dict                       # seg -> (aa_start, aa_end) over fully-owned residues, or None
    phase: dict                           # seg -> {"start": 0|1|2, "end": 0|1|2}


def project_codons(seq_len_nt: int, frame_offset: int,
                   segments: dict) -> AAProjection:
    """Assign each residue of the in-frame translation a segment provenance.

    ``segments`` maps seg name -> half-open ``(start, end)`` nt coords in the assembled read.
    """
    n_res = max(0, (seq_len_nt - frame_offset) // 3)
    prov = []
    for r in range(n_res):
        a = frame_offset + 3 * r
        b = a + 3
        owner = None
        for seg, (s, e) in segments.items():
            if s <= a and b <= e:
                owner = seg
                break
        prov.append(owner)
    aa_bounds = {}
    for seg in segments:
        idx = [r for r, o in enumerate(prov) if o == seg]
        aa_bounds[seg] = (idx[0], idx[-1] + 1) if idx else None
    phase = {seg: {"start": (s - frame_offset) % 3, "end": (e - frame_offset) % 3}
             for seg, (s, e) in segments.items()}
    return AAProjection(n_res, tuple(prov), aa_bounds, phase)
