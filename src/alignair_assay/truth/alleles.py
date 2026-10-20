"""Observable allele sets: the alleles indistinguishable given the germline actually visible in the read."""
from __future__ import annotations

from dataclasses import dataclass

from alignair_assay.core.reference import Reference, gene_name
from alignair_assay.truth.segment import ObservedSegment


@dataclass(frozen=True)
class AlleleTruth:
    """Truth for one allele call.

    ``primary`` is the generative allele, the one the simulator sampled (GenAIRR's ``truth_{g}_call``). A record
    that does not name it falls back to the first entry of the simulator's call list (``primary_fallback``).
    ``generative`` is the primary's observable set: every reference allele whose sequence contains the observed
    germline. ``alleles``, the scored set, adds the simulator's call list (``simulator_ties``, GenAIRR's
    post-SHM best-match re-call), restricted to reference alleles: both are defensible calls for a sequence-only
    aligner. ``disagrees_with_simulator`` is True when the simulator's call list is not exactly ``generative``
    (including an empty list or names not in the reference); most such lists are ties that hold the generative
    allele. ``recalled`` is the narrower case where the simulator called another allele first.
    """

    primary: str
    alleles: frozenset[str]
    genes: frozenset[str]
    simulator_ties: frozenset[str]
    disagrees_with_simulator: bool
    generative: frozenset[str]
    primary_fallback: bool = False

    @property
    def recalled(self) -> bool:
        """True when the simulator's first call is not the generative allele: GenAIRR re-called the read as
        another allele after SHM. GenAIRR lists the generative allele first whenever it lists it, so this is a
        non-empty call list without the primary. Never true for a fallback primary, which is the first call."""
        return not self.primary_fallback and bool(self.simulator_ties) and self.primary not in self.simulator_ties


def observable_alleles(seg: ObservedSegment, reference: Reference, gene_letter: str,
                       simulator_calls: list[str], *, primary_fallback: bool = False) -> AlleleTruth:
    """Truth for the allele of ``seg`` (``seg.allele`` is the primary): its observable set (alleles whose sequence
    contains the observed germline) and the scored set, that set plus the simulator's calls.

    Simulator names absent from ``reference`` are left out of the scored set but still count as a disagreement.
    ``primary_fallback`` records that the primary is the simulator's first call, not a named generative allele.
    """
    observed = seg.observed_germline.upper()
    if not observed:
        raise ValueError("observed germline is empty; every allele would match")
    matching = {name for name, seq in reference.alleles[gene_letter].items() if observed in seq}
    matching.add(seg.allele)
    generative = frozenset(matching)
    simulator = frozenset(simulator_calls)
    alleles = generative | {a for a in simulator if reference.has(a)}
    return AlleleTruth(
        primary=seg.allele,
        alleles=alleles,
        genes=frozenset(gene_name(a) for a in alleles),
        simulator_ties=simulator,
        disagrees_with_simulator=generative != simulator,
        generative=generative,
        primary_fallback=primary_fallback,
    )
