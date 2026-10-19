"""Observable allele sets: the alleles indistinguishable given the germline actually visible in the read."""
from __future__ import annotations

from dataclasses import dataclass

from alignair_assay.core.reference import Reference, gene_name
from alignair_assay.truth.segment import ObservedSegment


@dataclass(frozen=True)
class AlleleTruth:
    """Truth for one allele call.

    ``disagrees_with_simulator`` is True when the simulator list is empty or contains names not in the
    reference, as well as when the two sets differ.
    """

    primary: str
    alleles: frozenset[str]
    genes: frozenset[str]
    simulator_ties: frozenset[str]
    disagrees_with_simulator: bool


def observable_alleles(seg: ObservedSegment, reference: Reference, gene_letter: str,
                       simulator_calls: list[str]) -> AlleleTruth:
    """Alleles whose sequence contains the observed germline, plus the simulator's own calls.

    Simulator names absent from ``reference`` are left out of the set but still count as a disagreement.
    """
    observed = seg.observed_germline.upper()
    if not observed:
        raise ValueError("observed germline is empty; every allele would match")
    matching = {name for name, seq in reference.alleles[gene_letter].items() if observed in seq}
    matching.add(seg.allele)
    simulator = frozenset(simulator_calls)
    alleles = frozenset(matching | {a for a in simulator if reference.has(a)})
    return AlleleTruth(
        primary=seg.allele,
        alleles=alleles,
        genes=frozenset(gene_name(a) for a in alleles),
        simulator_ties=simulator,
        disagrees_with_simulator=frozenset(matching) != simulator,
    )
