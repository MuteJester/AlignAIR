"""A curriculum stream may draw its reads from novel alleles (``train.gym.novel``): each epoch of a stream's seeds
(``epoch_records`` seeds) makes a share of the eligible V genes homozygous for SNP variants drawn from the epoch alone,
so record k of seed s stays record k - a of seed s + a (worker plans and cache segments join); a read from a novel
allele names it and carries its definition. Off by default: the mix's streams are as before."""
import dataclasses
from itertools import islice

import GenAIRR.data as gd
import pytest

from alignair.errors import ConfigurationError
from alignair.reference.reference_set import ReferenceSet
from alignair.reference.variants import NOVEL_FIELD, eligible_bases
from alignair.train.records.novel import NovelAlleles, epoch_alleles, novel_support
from alignair.train.records.seeds import seed_block
from alignair.train.records.streams import CurriculumStreamPolicy, stream_records

SPEC = NovelAlleles()


def _streams(policy):
    return [stream for component in policy.substreams_for(gd.HUMAN_IGH_OGRDB) for stream in component]


@pytest.mark.component
def test_the_mix_without_novel_alleles_keeps_its_parameters():
    from alignair_lab.recipe import POLICY
    for policy in (CurriculumStreamPolicy(), POLICY):
        assert not any("novel_alleles" in parameters for _, parameters in _streams(policy))


@pytest.mark.component
def test_novel_alleles_reach_every_stream_of_the_recipes_mix():
    from alignair_lab.recipe import POLICY
    streams = _streams(dataclasses.replace(POLICY, novel_alleles=SPEC))
    assert len(streams) == 28
    assert all(parameters["novel_alleles"] == {"generator": 1, "gene_fraction": 0.4, "snps": [1, 6],
                                               "epoch_records": 4096} for _, parameters in streams)


@pytest.mark.component
@pytest.mark.parametrize("field, value", [("gene_fraction", 0), ("gene_fraction", 1.5), ("gene_fraction", True),
                                          ("snps", (0, 2)), ("snps", (3, 2)), ("snps", (1, 11)),
                                          ("epoch_records", 1000), ("epoch_records", 0), ("epoch_records", 2 ** 33),
                                          ("epoch_records", True)])
def test_an_invalid_novel_allele_spec_is_refused(field, value):
    with pytest.raises(ConfigurationError) as caught:
        NovelAlleles(**{field: value})
    assert caught.value.context["field"] == field


@pytest.mark.component
def test_a_stream_of_another_generator_is_refused():
    with pytest.raises(ConfigurationError):
        NovelAlleles.from_dict({**SPEC.to_dict(), "generator": 2})
    assert NovelAlleles.from_dict(SPEC.to_dict()) == SPEC
    with pytest.raises(ConfigurationError):
        CurriculumStreamPolicy(novel_alleles=SPEC.to_dict())


@pytest.mark.component
def test_an_epochs_alleles_are_drawn_from_the_epoch_alone():
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    alleles = epoch_alleles(reference, SPEC, 7)
    assert alleles == epoch_alleles(reference, SPEC, 7) and alleles != epoch_alleles(reference, SPEC, 8)
    assert len(alleles) == 20 and len({a.name.split("*")[0] for a in alleles}) == 20
    eligible = {a for names in eligible_bases(reference).values() for a in names}
    v = reference.gene("V")
    for allele in alleles:
        base = v.sequences[v.index[allele.base]]
        assert allele.base in eligible and 1 <= sum(x != y for x, y in zip(base, allele.sequence)) <= 6


def _seed(policy, component, stream):
    return seed_block(0, "training", 0, 0, component, stream)


@pytest.mark.integration
def test_reads_from_novel_alleles_name_them_and_carry_their_definition():
    policy = CurriculumStreamPolicy(novel_alleles=SPEC)
    name = "unselected-0"
    parameters = dict(policy.substreams_for(gd.HUMAN_IGH_OGRDB)[1])[name]
    seed = _seed(policy, 1, name)
    records = list(islice(stream_records(gd.HUMAN_IGH_OGRDB, parameters, seed), 1024))
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    drawn = {a.name for a in epoch_alleles(reference, SPEC, seed // SPEC.epoch_records)}
    novel = [r for r in records if NOVEL_FIELD in r]
    assert 0.06 <= len(novel) / len(records) <= 0.22
    for r in novel:
        assert r["v_call"] == r[NOVEL_FIELD]["V"]["name"] and r["v_call"] in drawn
        assert r["recalled_v_call"] == r["v_call"]      # so the allele-set target adds no base allele
    assert not any(r["v_call"] in drawn for r in records if NOVEL_FIELD not in r)
    assert [r["sequence_id"] for r in records] == [f"seq{k}" for k in range(len(records))]


@pytest.mark.integration
@pytest.mark.parametrize("stream, count, offsets", [("unselected-0", 24, (3, 8, 13)), ("selected-0", 12, (5, 8))])
def test_a_novel_stream_is_shiftable_across_an_epoch_boundary(stream, count, offsets):
    spec = NovelAlleles(epoch_records=8)
    policy = CurriculumStreamPolicy(novel_alleles=spec)
    parameters = dict(policy.substreams_for(gd.HUMAN_IGH_OGRDB)[0])[stream]
    base = _seed(policy, 0, stream) // 8 * 8                     # an epoch's first seed
    first = list(islice(stream_records(gd.HUMAN_IGH_OGRDB, parameters, base), count))
    for offset in offsets:
        shifted = list(islice(stream_records(gd.HUMAN_IGH_OGRDB, parameters, base + offset), count - offset))
        assert [{**r, "sequence_id": f"seq{offset + i}"} for i, r in enumerate(shifted)] == first[offset:]
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    epochs = [epoch_alleles(reference, spec, base // 8 + e) for e in range(3)]
    assert len({tuple(a.name for a in alleles) for alleles in epochs}) == 3


@pytest.mark.integration
def test_novel_support_refuses_a_dataconfig_without_an_eligible_v():
    with pytest.raises(ConfigurationError, match="MOUSE_TCRA_IMGT"):
        novel_support(gd.MOUSE_TCRA_IMGT, SPEC)
    assert novel_support(gd.HUMAN_IGH_OGRDB, SPEC) == 51
