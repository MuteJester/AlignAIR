"""Translated V/J germlines with conserved-anchor QC.

D is intentionally excluded: a D allele has no single canonical protein form (it appears
in any frame, is trimmed both ends, and its boundary codons absorb N-region nucleotides).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .translation import translate_nt

ANCHOR_AA: dict[str, str] = {"V": "C", "J": "W"}   # Cys-104 (V), Trp-118 (J)


@dataclass(frozen=True)
class AAGermline:
    name: str
    aa_seq: str        # in-frame translation of the ungapped germline
    frame: int         # anchor % 3
    anchor_aa: int     # residue index of the conserved anchor
    gene: str          # "V" | "J"


@dataclass
class AAGermlineSet:
    genes: dict = field(default_factory=dict)         # gene -> {name: AAGermline}
    quarantined: dict = field(default_factory=dict)   # gene -> {name: reason}


def translate_germline(name: str, ungapped_seq: str, anchor: int, gene: str):
    """Translate one germline in its anchor frame; QC the conserved anchor residue.

    Returns ``(AAGermline, None)`` on pass, or ``(None, reason)`` if the anchor residue is
    absent/wrong (bad frame or pseudogene) -> quarantine.
    """
    expected = ANCHOR_AA[gene]
    frame = anchor % 3
    aa = translate_nt(ungapped_seq, frame)
    anchor_aa = (anchor - frame) // 3
    if anchor_aa >= len(aa):
        return None, f"anchor residue index {anchor_aa} out of range (len {len(aa)})"
    if aa[anchor_aa] != expected:
        return None, f"anchor residue {aa[anchor_aa]!r} != expected {expected!r}"
    return AAGermline(name, aa, frame, anchor_aa, gene), None


def build_aa_germlines(dc, genes=("V", "J")) -> AAGermlineSet:
    """Build translated V/J germlines from a GenAIRR dataconfig, quarantining anchor-QC failures."""
    out: dict[str, dict] = {}
    quar: dict[str, dict] = {}
    for g in genes:
        gU = g.upper()
        out[gU], quar[gU] = {}, {}
        for allele in dc.allele_list(g.lower()):
            if allele.anchor is None:
                quar[gU][allele.name] = "no conserved anchor"
                continue
            germ, reason = translate_germline(
                allele.name, allele.ungapped_seq.upper(), int(allele.anchor), gU)
            if germ is None:
                quar[gU][allele.name] = reason
            else:
                out[gU][allele.name] = germ
    return AAGermlineSet(out, quar)
