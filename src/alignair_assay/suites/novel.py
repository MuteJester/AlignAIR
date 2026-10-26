"""Novel alleles: functional variants of a reference allele, absent from the reference every tool is given.

A novel allele carries SNPs only at its base's invariant positions, where every other allele of the base's gene
agrees with the base, and never in the conserved anchor codon. Compared position by position over any germline
span, another allele of the gene that covers the span differs from the novel allele at every SNP in it and
wherever it differs from the base, so the base is strictly the nearest allele of its gene (an equally near one
would equal the base over the span, and so be indistinguishable from it). A perfect aligner restricted to the
reference therefore reports the base, which is how the truth of a novel read is built (``truth.build``).

GenAIRR's ``Genotype.add_novel_allele`` is the validity oracle: it rejects a variant whose anchor codon no longer
encodes the conserved residue or whose coding frame gains a stop codon.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

from alignair_assay.core.genotype import NovelAllele
from alignair_assay.core.reference import Reference, gene_name

if TYPE_CHECKING:
    from GenAIRR import Genotype

# Draws of SNP positions and bases ``make_novel_allele`` tries before it gives up.
MAX_ATTEMPTS = 50
_BASES = "ACGT"


def invariant_positions(reference: Reference, base: str) -> list[int]:
    """The 0-based germline positions ``p < len(base)`` of reference allele ``base`` at which every other allele of
    its gene that is longer than ``p`` has the same nucleotide, in increasing order; the positions of the base's
    anchor codon ``[anchor, anchor + 3)`` are left out (none for an allele without an anchor). KeyError when
    ``base`` is not in ``reference``."""
    letter = reference.gene_letter(base)
    sequence = reference.alleles[letter][base]
    gene = gene_name(base)
    others = [seq for name, seq in reference.alleles[letter].items() if name != base and gene_name(name) == gene]
    anchor = reference.anchors.get(base)
    codon = range(anchor, anchor + 3) if anchor is not None else range(0)
    return [p for p, nucleotide in enumerate(sequence)
            if p not in codon and all(len(other) <= p or other[p] == nucleotide for other in others)]


def make_novel_allele(reference: Reference, base: str, k: int, rng: random.Random, name: str, *,
                      genotype: Genotype) -> NovelAllele:
    """A novel allele ``name`` derived from reference allele ``base`` by ``k`` SNPs: ``k`` distinct invariant
    positions (``invariant_positions``) drawn from ``rng``, each given a different nucleotide drawn uniformly.

    ``genotype`` is a GenAIRR ``Genotype`` of the DataConfig ``reference`` was built from; its
    ``add_novel_allele`` validates each draw, and the accepted allele stays registered on it under ``name`` (so
    one genotype validates one donor's alleles, whose names differ). A rejected draw is redrawn, up to
    ``MAX_ATTEMPTS`` draws in all; RuntimeError, naming the base and the last rejection, when every draw is
    rejected. ValueError when ``k`` is not between 1 and the number of invariant positions.
    """
    sequence = reference.seq(base)
    positions = invariant_positions(reference, base)
    if not 1 <= k <= len(positions):
        raise ValueError(f"{base}: cannot place {k} SNPs at {len(positions)} invariant positions")
    segment = reference.gene_letter(base).upper()
    rejection = None
    for _ in range(MAX_ATTEMPTS):
        snps = sorted(rng.sample(positions, k))
        variant = list(sequence)
        for p in snps:
            variant[p] = rng.choice([b for b in _BASES if b != sequence[p]])
        candidate = "".join(variant)
        try:
            genotype.add_novel_allele(name, base=base, sequence=candidate, segment=segment)
        except ValueError as exc:
            rejection = exc
            continue
        return NovelAllele(name=name, base=base, sequence=candidate, snps=tuple(snps))
    raise RuntimeError(f"no novel allele of {base} with {k} SNPs was accepted in {MAX_ATTEMPTS} draws; "
                       f"last rejection: {rejection}")
