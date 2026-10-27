"""Explicit classification grouping policies, independent of prediction execution."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..errors import ConfigurationError


@dataclass(frozen=True)
class ClassificationStratum:
    """Group a supervised singleton truth row using its original record and true class ID.

    ``group(record, true_class)`` returns a nonempty string label. Coordinate units and any
    reference/class-map binding belong to this policy; callbacks must not mutate records.
    """
    name: str
    gene: str
    group: Callable[[Any, int], str]

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name.strip()
                or not isinstance(self.gene, str) or not self.gene.strip() or not callable(self.group)):
            raise ConfigurationError("classification strata require a name, gene, and callable grouping policy")


def protein_v_strata(class_map) -> tuple[ClassificationStratum, ...]:
    """Established protein-study V bins: visible residue span and synonymous allele count.

    Records must use residue-space V boundaries. This preset does not translate nucleotide
    coordinates; use caller-defined strata for other coordinate units or scientific policies.
    """
    if not callable(getattr(class_map, "ambiguity_set", None)):
        raise ConfigurationError("protein V strata require a class map with ambiguity_set")

    def coverage(record, true_class):
        span = record["v_end"] - record["v_start"]
        return "<40" if span < 40 else ("40-79" if span < 80 else ">=80")

    def ambiguity(record, true_class):
        count = len(class_map.ambiguity_set(true_class))
        return str(count) if count < 3 else ">=3"

    return (ClassificationStratum("v_acc_by_coverage", "v", coverage),
            ClassificationStratum("v_acc_by_ambiguity_size", "v", ambiguity))
