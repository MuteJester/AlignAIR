"""Amino-acid translation + codon-provenance projection.

Coordinate projection is NOT ``//3``: GenAIRR translates the assembled read in a
junction-derived reading frame and drops incomplete boundary codons. A residue is
assigned to a nucleotide segment only when ALL THREE of its codon's nucleotides lie
inside that segment; codons that straddle a segment boundary (or land in an N/P gap)
are masked (provenance ``None``).
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from operator import index
from numbers import Integral

from .coordinates import query_spans
from ..errors import TrainingDataError


def projection_segments(record):
    """Query-only segment view; codon projection does not require germline alignment fields."""
    return {gene.upper(): span for gene, span in query_spans(record, include_empty=True).items()}


def residue_count(seq_len_nt: int, frame_offset: int) -> int:
    """Number of complete codons in the declared read frame."""
    return max(0, (seq_len_nt - frame_offset) // 3)


def protein_translation_issue(record) -> str | None:
    """Return an eligibility reason; malformed annotation types raise rather than becoming rejections."""
    if not isinstance(record, Mapping):
        raise TrainingDataError("protein records must be mappings")
    aa = record.get("sequence_aa")
    if aa is not None and not isinstance(aa, str):
        raise TrainingDataError("protein sequence must be a string", context={"field": "sequence_aa"})
    if not aa:
        return "no_aa"
    if record.get("junction_start") is None:
        return "no_frame"
    if not isinstance(record.get("sequence"), str):
        raise TrainingDataError("nucleotide sequence must be a string", context={"field": "sequence"})
    try:
        junction = record["junction_start"]
        if isinstance(junction, bool) or not isinstance(junction, Integral):
            raise TypeError
        frame = index(junction) % 3
    except TypeError as exc:
        raise TrainingDataError("junction frame requires an integer coordinate",
                                context={"field": "junction_start"}) from exc
    if len(record["sequence_aa"]) != residue_count(len(record["sequence"]), frame):
        return "translation_length_mismatch"
    return None


def translated_record_usable(record) -> bool:
    """Whether the exported protein length agrees with its declared nucleotide reading frame."""
    return protein_translation_issue(record) is None

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
    n_res = residue_count(seq_len_nt, frame_offset)
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
