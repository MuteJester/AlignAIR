"""GenAIRR simulates reads from the novel alleles AlignAIR draws (``alignair.genairr.novel``): a permissive genotype
with each one homozygous for its gene; a GenAIRR refusal names the allele; ``derive_variant`` draws SNP variants."""
import random

import GenAIRR.data as gd
import pytest

from alignair.errors import ConfigurationError
from alignair.genairr.novel import derive_variant, novel_genotype
from alignair.reference.reference_set import ReferenceSet
from alignair.reference.variants import NovelAllele, eligible_bases, novel_name, snp_variant

pytestmark = pytest.mark.component


@pytest.fixture(scope="module")
def igh():
    return ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)


def test_genairr_accepts_every_variant_the_generator_draws(igh):
    eligible = eligible_bases(igh)
    v = igh.gene("V")
    for seed in range(4):
        rng = random.Random(seed)
        drawn = [variant for gene in sorted(eligible)
                 if (variant := snp_variant(igh, rng.choice(eligible[gene]), rng.randint(1, 6), rng)) is not None]
        genotype = novel_genotype(gd.HUMAN_IGH_OGRDB, drawn)
        assert genotype.novel_allele_names() == {a.name for a in drawn}
        effective = genotype.effective_dataconfig()
        for allele in drawn[:3]:
            entry = effective.v_alleles[allele.name.split("*")[0]][-1]
            assert entry.name == allele.name and entry.ungapped_seq.upper() == allele.sequence
            assert entry.anchor == v.anchors[allele.base]


def test_a_genairr_refusal_names_the_allele(igh):
    base = sorted(a for names in eligible_bases(igh).values() for a in names)[0]
    sequence = igh.gene("V").sequences[igh.gene("V").index[base]] + "A"
    name = novel_name(base, sequence)
    with pytest.raises(ConfigurationError) as caught:
        novel_genotype(gd.HUMAN_IGH_OGRDB, [NovelAllele(name, base, sequence)])
    assert caught.value.context["allele"] == name


def test_derive_variant_is_the_snp_generator(igh):
    base = sorted(a for names in eligible_bases(igh).values() for a in names)[0]
    assert derive_variant(igh, base, substitutions=2, rng=random.Random(3)) == snp_variant(igh, base, 2,
                                                                                         random.Random(3))
