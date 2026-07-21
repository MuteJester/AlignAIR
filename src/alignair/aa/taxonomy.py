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


def asc_to_imgt_map(dc) -> dict:
    """OGRDB ASC allele name (e.g. ``IGHVF1-G1*01``) -> its IMGT allele of origin.

    Sourced from the dataconfig's authoritative provenance table (``dc.asc_tables["V"]``), a list of
    dicts each carrying ``new_allele`` (the ASC name) and ``imgt_allele`` (the IMGT allele it derives
    from). Required before any family/gene scoring against IMGT-labelled gold data, since a model's
    ASC-named class map cannot be parsed as IMGT nomenclature directly (see ``family_of``).
    """
    return {row["new_allele"]: row["imgt_allele"] for row in dc.asc_tables["V"]}


class ImgtView:
    """View over an ASC (OGRDB) equivalence-class map with ``class_members`` translated to IMGT names.

    Exposes only ``class_members``, the sole attribute the taxonomy/scorer helpers read off a class
    map, so it is a drop-in substitute for a real class map wherever family/gene scoring happens.
    """

    def __init__(self, cmap, asc2imgt: dict):
        self.class_members = {
            cid: [asc2imgt[a] for a in members] for cid, members in cmap.class_members.items()
        }
