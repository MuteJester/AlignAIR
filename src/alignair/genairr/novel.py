"""GenAIRR's side of novel alleles: drawing a variant (``derive_variant``, the hook for GenAIRR's own generator) and
the genotype a stream simulates reads from (``novel_genotype``).

``derive_variant`` draws SNP variants with AlignAIR's generator (``alignair.reference.variants.snp_variant``). It is
the one place to call a GenAIRR variant generator (seeded SNP and short in-frame indel variants) should GenAIRR ship
one: with ``protect="anchor"``, ``seed=rng`` and ``distinct_from`` the dataconfig, passing the variant's gapped form
and anchor as supplied. A variant of another length also needs gapping by template, which
``alignair.reference.annotate`` has as its gapping hook."""
from __future__ import annotations

from typing import Sequence

from ..errors import ConfigurationError
from ..reference.variants import NovelAllele, snp_variant


def derive_variant(reference, base: str, *, substitutions: int, rng) -> NovelAllele | None:
    """A novel allele of ``base`` with ``substitutions`` SNPs (``snp_variant``)."""
    return snp_variant(reference, base, substitutions, rng)


def novel_genotype(dataconfig, alleles: Sequence[NovelAllele]):
    """A permissive genotype of ``dataconfig`` (every allele usable, as without one) in which each novel allele is
    homozygous for its gene, so the gene's reads come from it. ConfigurationError naming an allele GenAIRR refuses."""
    from GenAIRR import Genotype
    genotype = Genotype.permissive(dataconfig)
    for allele in alleles:
        try:
            genotype.add_novel_allele(allele.name, base=allele.base, sequence=allele.sequence, segment=allele.gene)
            genotype.homozygous(allele.name.split("*")[0], allele.name, segment=allele.gene)
        except ValueError as exc:
            raise ConfigurationError(f"GenAIRR refuses the novel allele {allele.name}: {exc}",
                                     context={"allele": allele.name}) from exc
    return genotype
