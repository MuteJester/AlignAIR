"""Allele naming and lossless class expansion, independent of model input alphabet."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
import re

from ..errors import ConfigurationError

_FAMILY = re.compile(r"^(IGH[VDJ]\d+)")


def _invalid(message, field):
    return ConfigurationError(message, context={"stage": "reference_nomenclature", "field": field})


def gene_of(allele: str) -> str:
    """Remove an allele suffix; preserve the gene name and its naming scheme."""
    if not isinstance(allele, str) or not allele or not allele.split("*", 1)[0]:
        raise _invalid("allele names must be nonempty strings with a gene name", "allele")
    return allele.split("*", 1)[0]


def family_of(allele: str) -> str:
    """Parse a numeric IMGT IGHV/IGHD/IGHJ family; ASC cluster names require explicit mapping."""
    match = _FAMILY.match(allele) if isinstance(allele, str) else None
    if match is None:
        raise _invalid(f"cannot parse IMGT family from {allele!r}", "allele")
    return match.group(1)


def _names(values, field):
    if (not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values
            or any(not isinstance(name, str) or not name for name in values)):
        raise _invalid("allele members must be a nonempty sequence of names", field)
    return values


def asc_to_imgt_multi(dataconfig) -> dict[str, list[str]]:
    """Retain every provenance mapping, sorted and deduplicated; only V may be ASC-clustered."""
    tables = getattr(dataconfig, "asc_tables", None)
    if not isinstance(tables, Mapping) or "V" not in tables:
        raise _invalid("ASC provenance requires an asc_tables V table", "asc_tables")
    if set(tables) != {"V"}:
        raise _invalid("ASC expansion assumes only V is ASC-clustered", "asc_tables")
    rows = tables["V"]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise _invalid("ASC provenance rows must be a sequence of mappings", "asc_tables.V")
    expanded = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise _invalid("ASC provenance rows must be mappings", "asc_tables.V")
        source, target = row.get("new_allele"), row.get("imgt_allele")
        if any(not isinstance(name, str) or not name for name in (source, target)):
            raise _invalid("ASC provenance requires nonempty new_allele and imgt_allele names", "asc_tables.V")
        expanded.setdefault(source, set()).add(target)
    return {source: sorted(targets) for source, targets in expanded.items()}


def class_alleles(class_id, class_map, allele_mapping=None) -> list[str]:
    """Expand every class member; unmapped names retain their identity and unknown IDs raise KeyError."""
    if isinstance(class_id, bool) or not isinstance(class_id, Integral) or class_id < 0:
        raise _invalid("class_id must be a nonnegative integer", "class_id")
    members = getattr(class_map, "class_members", None)
    if not isinstance(members, Mapping):
        raise _invalid("class map must supply class_members", "class_members")
    if allele_mapping is not None and not isinstance(allele_mapping, Mapping):
        raise _invalid("allele expansion must be a name-to-members mapping", "allele_mapping")
    expanded = set()
    for member in _names(members[class_id], "class_members"):
        targets = (member,) if allele_mapping is None else allele_mapping.get(member, (member,))
        expanded.update(_names(targets, "allele_mapping"))
    return sorted(expanded)


def class_genes(class_id, class_map, allele_mapping=None) -> list[str]:
    return sorted({gene_of(name) for name in class_alleles(class_id, class_map, allele_mapping)})


def class_families(class_id, class_map, allele_mapping=None) -> list[str]:
    return sorted({family_of(name) for name in class_alleles(class_id, class_map, allele_mapping)})


def catalogue_families(class_map, allele_mapping=None) -> list[str]:
    """Sorted distinct families across all classes under the same explicit expansion policy."""
    members = getattr(class_map, "class_members", None)
    if not isinstance(members, Mapping):
        raise _invalid("class map must supply class_members", "class_members")
    return sorted({family for class_id in members for family in class_families(class_id, class_map, allele_mapping)})
