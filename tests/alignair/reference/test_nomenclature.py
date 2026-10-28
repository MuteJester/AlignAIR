"""Reference naming and class expansion preserve complete, deterministic ambiguity sets."""
from types import SimpleNamespace

import pytest

from alignair.errors import ConfigurationError
from alignair.reference import (asc_to_imgt_multi, catalogue_families, class_alleles,
                                class_genes, class_families, family_of, gene_of)

pytestmark = pytest.mark.component

class FakeDC:
    asc_tables = {"V": [
        {"new_allele": "IGHVF1-G1*01", "imgt_allele": "IGHV3-23*01"},
        {"new_allele": "IGHVF1-G1*01", "imgt_allele": "IGHV3-23D*01"},  # SAME asc -> two IMGT (one-to-many)
        {"new_allele": "IGHVF1-G2*01", "imgt_allele": "IGHV1-2*02"},
    ]}
class FakeMap:
    class_members = {0: ["IGHVF1-G1*01"], 1: ["IGHVF1-G1*01", "IGHVF1-G2*01"]}

def test_one_to_many_retains_all():
    m = asc_to_imgt_multi(FakeDC())
    assert m["IGHVF1-G1*01"] == ["IGHV3-23*01", "IGHV3-23D*01"]   # BOTH kept, sorted


def test_class_alleles_union_all_members():
    m = asc_to_imgt_multi(FakeDC())
    assert class_alleles(1, FakeMap(), m) == ["IGHV1-2*02", "IGHV3-23*01", "IGHV3-23D*01"]
    assert class_genes(1, FakeMap(), m) == ["IGHV1-2", "IGHV3-23", "IGHV3-23D"]
    assert class_families(0, FakeMap(), m) == ["IGHV3"]


def test_asc_to_imgt_multi_raises_on_extra_asc_keys():
    """asc_to_imgt_multi assumes only V is ASC-clustered; raise ValueError if D/J present."""
    class BadDC:
        asc_tables = {"V": [], "D": []}

    with pytest.raises(ValueError, match="only V is ASC-clustered"):
        asc_to_imgt_multi(BadDC())


@pytest.mark.parametrize('allele,gene,family', [
    ('IGHV3-23*01', 'IGHV3-23', 'IGHV3'), ('IGHV1-69D*01', 'IGHV1-69D', 'IGHV1'),
    ('IGHD2-2*01', 'IGHD2-2', 'IGHD2'), ('IGHJ6*02', 'IGHJ6', 'IGHJ6'),
])
def test_gene_and_imgt_family_parsing(allele, gene, family):
    assert gene_of(allele) == gene and family_of(allele) == family


def test_unmapped_names_use_identity_expansion_and_shared_class_taxonomy():
    cmap = SimpleNamespace(class_members={0: ['IGHV3-23*01', 'IGHV3-23*02'], 1: ['IGHV1-69D*01']})
    assert class_alleles(0, cmap) == ['IGHV3-23*01', 'IGHV3-23*02']
    assert class_genes(0, cmap) == ['IGHV3-23']
    assert class_families(0, cmap) == ['IGHV3']
    assert catalogue_families(cmap) == ['IGHV1', 'IGHV3']
    assert catalogue_families(SimpleNamespace(class_members={})) == []


def test_mapping_keeps_cross_family_origins_and_deduplicates_without_mutating_inputs():
    dc = SimpleNamespace(asc_tables={'V': [
        {'new_allele': 'IGHVF1-G1*01', 'imgt_allele': 'IGHV3-23*01'},
        {'new_allele': 'IGHVF1-G1*01', 'imgt_allele': 'IGHV2-70*04'},
        {'new_allele': 'IGHVF1-G1*01', 'imgt_allele': 'IGHV3-23*01'},
    ]})
    mapping = asc_to_imgt_multi(dc)
    assert mapping == {'IGHVF1-G1*01': ['IGHV2-70*04', 'IGHV3-23*01']}
    assert class_families(0, FakeMap(), mapping) == ['IGHV2', 'IGHV3']
    assert catalogue_families(SimpleNamespace(class_members={0: ['IGHVF1-G1*01']}), mapping) == ['IGHV2', 'IGHV3']
    names = class_alleles(0, FakeMap(), mapping)
    names.clear()
    assert mapping['IGHVF1-G1*01'] == ['IGHV2-70*04', 'IGHV3-23*01']


@pytest.mark.parametrize('allele', [None, '', '*01', 1])
def test_invalid_gene_names_have_reference_context(allele):
    with pytest.raises(ConfigurationError) as caught:
        gene_of(allele)
    assert caught.value.context['stage'] == 'reference_nomenclature'


@pytest.mark.parametrize('allele', [None, '', 'notagene', 'IGHVF1-G1*01'])
def test_family_parser_never_treats_asc_cluster_ids_as_imgt_families(allele):
    with pytest.raises(ConfigurationError):
        family_of(allele)


@pytest.mark.parametrize('tables', [None, {}, {'V': None}, {'V': ''}, {'V': [None]},
    {'V': [{}]}, {'V': [{'new_allele': '', 'imgt_allele': 'IGHV3-23*01'}]}])
def test_malformed_provenance_fails_with_configuration_errors(tables):
    with pytest.raises(ConfigurationError):
        asc_to_imgt_multi(SimpleNamespace(asc_tables=tables))


@pytest.mark.parametrize('class_id', [True, -1, 1.5, '0'])
def test_class_ids_are_explicit_nonnegative_integers(class_id):
    with pytest.raises(ConfigurationError):
        class_alleles(class_id, FakeMap())


@pytest.mark.parametrize('members', [None, 'IGHV3-23*01', [], [None]])
def test_class_members_cannot_silently_expand_to_characters_or_empty_labels(members):
    with pytest.raises(ConfigurationError):
        class_alleles(0, SimpleNamespace(class_members={0: members}))


@pytest.mark.parametrize('mapping', [[], {'IGHVF1-G1*01': 'IGHV3-23*01'}, {'IGHVF1-G1*01': []}])
def test_malformed_expansion_maps_fail_before_returning_an_allele_set(mapping):
    with pytest.raises(ConfigurationError):
        class_alleles(0, FakeMap(), mapping)


def test_unknown_ids_and_absent_class_maps_have_distinct_errors():
    with pytest.raises(KeyError):
        class_alleles(99, FakeMap())
    with pytest.raises(ConfigurationError):
        class_alleles(0, None)
    with pytest.raises(ConfigurationError):
        catalogue_families(None)
