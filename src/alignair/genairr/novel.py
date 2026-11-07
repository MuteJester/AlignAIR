"""GenAIRR's side of novel alleles: drawing a variant (``derive_variant``, the hook for GenAIRR's own generator) and
the genotype a stream simulates reads from (``novel_genotype``).

``derive_variant`` is AlignAIR's SNP generator (``alignair.reference.variants.snp_variant``) until GenAIRR ships
``GenAIRR.alleles.derive_variant`` (ask G3: seeded SNP and short in-frame indel variants). Then it calls that with
``protect="anchor"``, ``seed=rng`` and ``distinct_from`` the dataconfig, and passes the variant's gapped form and
anchor as supplied; a variant of another length also needs GenAIRR's gapping by template (ask G6), which
``alignair.reference.annotate`` has as its gapping hook."""
from __future__ import annotations

from typing import Sequence

from ..errors import ConfigurationError
from ..reference.variants import NovelAllele, snp_variant


def derive_variant(reference, base: str, *, substitutions: int, rng, indels: int = 0) -> NovelAllele | None:
    """A novel allele of ``base`` (``snp_variant``); short indels wait for GenAIRR's ``derive_variant`` (ask G3)."""
    if indels > 0:
        raise NotImplementedError("short indel novel alleles wait for GenAIRR's derive_variant (ask G3)")
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
