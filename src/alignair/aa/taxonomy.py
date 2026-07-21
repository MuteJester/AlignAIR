"""IMGT allele -> gene -> family canonicalisation + class expansion (no alignment)."""
from __future__ import annotations
import re
_FAM = re.compile(r"^(IGH[VDJ]\d+)")

def gene_of(allele: str) -> str:
    return allele.split("*")[0]

def family_of(name: str) -> str:
    m = _FAM.match(name)
    if not m:
        raise ValueError(f"cannot parse IMGT family from {name!r}")
    return m.group(1)

def class_genes(cid: int, cmap) -> set:
    return {gene_of(a) for a in cmap.class_members[cid]}

def class_families(cid: int, cmap) -> set:
    return {family_of(a) for a in cmap.class_members[cid]}

def catalogue_families(cmap) -> set:
    fams = set()
    for members in cmap.class_members.values():
        fams |= {family_of(a) for a in members}
    return fams
