"""Completing a reference against a known one: IMGT-gapped forms and conserved anchors for the alleles that lack them.

The model lays V on IMGT-gapped columns and lines a read up with an allele from its conserved codon
(V Cys, J W/F), so a user's reference (a FASTA, a genotype with novel alleles) is completed against the reference the
model came with (``known``) before it is bound. Each allele without a gapped form or anchor takes them from:

- the known allele of the same name, when its sequence is the same; else
- the known allele of the same gene type and length that differs from it at the fewest bases, when those are at most
  ``MAX_DIFFERENT`` of its bases (a SNP variant: its own bases on the known allele's gapped columns, and the known
  allele's anchor, at the same base).

An allele with neither (an indel variant, an unrelated sequence) stays without: the model then reads its own positions
as its columns and its alignment units skip it. A gapped form or anchor the reference already gives is kept.
"""
from __future__ import annotations

from .reference_set import GeneReference, ReferenceSet

MAX_DIFFERENT = 0.1          # the largest share of an allele's bases its source may differ at
GAP = "."


def _source(name: str, sequence: str, known: GeneReference | None) -> str | None:
    """The known allele an allele takes its gapped form and anchor from (see the module docstring), or None."""
    if known is None:
        return None
    if name in known.index and known.sequences[known.index[name]] == sequence:
        return name
    candidates = [(sum(a != b for a, b in zip(other, sequence)), other_name)
                  for other_name, other in zip(known.names, known.sequences) if len(other) == len(sequence)]
    if not candidates:
        return None
    different, best = min(candidates)
    return best if different <= MAX_DIFFERENT * len(sequence) else None


def _carry_gaps(gapped: str, sequence: str) -> str:
    """A known allele's gapped form with another allele's bases (the same length) in its base columns."""
    bases = iter(sequence)
    return "".join(GAP if column == GAP else next(bases) for column in gapped)


def complete_reference(reference: ReferenceSet, known: ReferenceSet) -> ReferenceSet:
    """``reference`` with every allele's missing gapped form and anchor taken from ``known`` where one is found."""
    genes = {}
    for gene, alleles in reference.genes.items():
        target = known.genes.get(gene)
        anchors, gapped = dict(alleles.anchors or {}), dict(alleles.gapped or {})
        for name, sequence in zip(alleles.names, alleles.sequences):
            if name in anchors and (name in gapped or gene != "V"):
                continue
            source = _source(name, sequence, target)
            if source is None:
                continue
            if name not in gapped and target.gapped and source in target.gapped:
                gapped[name] = _carry_gaps(target.gapped[source], sequence)
            if name not in anchors and target.anchors and source in target.anchors:
                anchors[name] = target.anchors[source]
        genes[gene] = GeneReference(list(alleles.names), list(alleles.sequences), dict(alleles.index),
                                    anchors=anchors or None, gapped=gapped or None)
    return ReferenceSet(genes, reference.has_d, reference.loci)
