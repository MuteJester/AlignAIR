"""The genotype pass: the donor's V genotype from a repertoire's own rows, and the rows it re-calls.

A single read cannot always tell two alleles apart (``v_call`` lists the ones it ties, ``predict.ties``); a repertoire
can, since the donor carries few of each gene's alleles. As TIgGER's frequency rule does on IgBLAST's output, the
genotype counts, per V gene, the reads whose tie list is a single allele of it, and keeps the most frequent alleles
until they explain ``EXPLAIN`` of those reads; a gene with fewer than ``MIN_READS`` of them has no evidence and stays
open (every allele allowed). A row then moves when its call is outside the genotype and its tie list holds an allowed
allele; a row whose read matches no allowed allele as well as its call keeps it. The CLI re-calls the moved rows with
the genotype as the ``--genotype`` constraint, so their rows stay consistent.

The assay's real arm (donors sequenced by RACE and by multiplex) measures how far the rule lowers the share of V calls
the donor's genotype does not support, beside the same rule on IgBLAST's own tie lists (about IgBLAST + TIgGER); the
AlignAIR 4 figures are pending (PENDING-V4-NUMBERS).
"""
from __future__ import annotations

from collections import Counter, defaultdict

EXPLAIN = 0.875          # the share of a gene's single-allele reads its genotype alleles explain (TIgGER's default)
MIN_READS = 10           # the single-allele reads a gene needs for its alleles to be inferred


def gene_of(allele: str) -> str:
    return allele.split("*")[0]


class GenotypeEvidence:
    """The rows' V calls, observed one by one (``observe``): the single-allele reads counted per gene, and the tie
    lists of the rows that list several alleles, by row index."""

    def __init__(self):
        self.counts: dict[str, Counter] = defaultdict(Counter)
        self.tied: dict[int, tuple[str, ...]] = {}

    def observe(self, index: int, v_call: str) -> None:
        listed = tuple(allele for allele in (v_call or "").split(",") if allele)
        if len(listed) == 1:
            self.counts[gene_of(listed[0])][listed[0]] += 1
        elif listed:
            self.tied[index] = listed

    def genotype(self, explain: float = EXPLAIN, min_reads: int = MIN_READS) -> dict[str, list[str]]:
        """``{gene: alleles}`` for the genes with evidence, each gene's alleles most frequent first."""
        out = {}
        for gene, counts in sorted(self.counts.items()):
            total = sum(counts.values())
            if total < min_reads:
                continue
            kept, explained = [], 0
            for allele, n in counts.most_common():
                if explained >= explain * total:
                    break
                kept.append(allele)
                explained += n
            out[gene] = kept
        return out

    def moved(self, genotype: dict[str, list[str]]) -> list[int]:
        """The indices of the rows whose call is outside ``genotype`` and whose tie list holds an allowed allele."""
        def allowed(allele):
            return gene_of(allele) not in genotype or allele in genotype[gene_of(allele)]
        return sorted(index for index, listed in self.tied.items()
                      if not allowed(listed[0]) and any(allowed(allele) for allele in listed[1:]))


def allowed_v(genotype: dict[str, list[str]], names) -> list[str]:
    """The V alleles of ``names`` (the reference's, in order) the genotype allows: a gene's genotype alleles where it
    has evidence, every allele of the others."""
    return [name for name in names if gene_of(name) not in genotype or name in genotype[gene_of(name)]]
