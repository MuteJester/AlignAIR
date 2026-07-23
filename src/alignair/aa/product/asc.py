"""One-to-many ASC->IMGT mapping + class expansion (retains every mapping; no representative)."""
from __future__ import annotations
import re
_FAM = re.compile(r"^(IGH[VDJ]\d+)")

def asc_to_imgt_multi(dc) -> dict:
    out: dict[str, set] = {}
    for row in dc.asc_tables["V"]:
        out.setdefault(row["new_allele"], set()).add(row["imgt_allele"])
    return {k: sorted(v) for k, v in out.items()}

def _gene(a): return a.split("*")[0]
def _fam(a):
    m = _FAM.match(a)
    if not m: raise ValueError(f"cannot parse family from {a!r}")
    return m.group(1)

def class_alleles(class_id, cmap, asc_multi) -> list:
    out = set()
    for member in cmap.class_members[class_id]:
        out.update(asc_multi.get(member, [member]))
    return sorted(out)

def class_genes(class_id, cmap, asc_multi) -> list:
    return sorted({_gene(a) for a in class_alleles(class_id, cmap, asc_multi)})

def class_families(class_id, cmap, asc_multi) -> list:
    return sorted({_fam(a) for a in class_alleles(class_id, cmap, asc_multi)})
