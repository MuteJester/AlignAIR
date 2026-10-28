"""One preflight contract for prepared, saved and restored reference-bound configurations."""
from copy import deepcopy
from dataclasses import replace

import pytest

from alignair.core import ModelSetup, prepare_model, validate_reference_binding
from alignair.errors import ConfigurationError, DataContractError
from alignair.reference import ReferenceSet

pytestmark = pytest.mark.component


def _setup(input_space='nucleotide', label_space='alleles', has_d=True):
    reference = ReferenceSet.from_genotype({
        'V': {'V2': 'ATGTGTAAA', 'V1': 'ATGTGCAAA', 'V3': 'ATGTGTTTT'},
        'D': {'D1': 'TAC'}, 'J': {'J2': 'TGGAAA', 'J1': 'TGGTTT'},
    })
    reference.gene('V').anchors = {name: 3 for name in reference.gene('V').names}
    reference.gene('J').anchors = {name: 0 for name in reference.gene('J').names}
    return prepare_model(reference, input_space=input_space, label_space=label_space, has_d=has_d)


@pytest.mark.parametrize('input_space', ['nucleotide', 'protein'])
@pytest.mark.parametrize('has_d', [False, True])
def test_allele_binding_needs_no_schema_and_allows_an_omitted_d_head(input_space, has_d):
    setup = _setup(input_space, has_d=has_d)
    before = setup.reference.to_serializable()
    validate_reference_binding(setup.reference, setup.config)
    assert setup.reference.to_serializable() == before


@pytest.mark.parametrize('gene', ['v', 'd', 'j'])
def test_every_enabled_head_must_match_its_reference_labels_without_a_schema(gene):
    setup = _setup()
    cfg = replace(setup.config, **{f'{gene}_allele_count': 9})
    with pytest.raises(ConfigurationError, match='head size') as caught:
        validate_reference_binding(setup.reference, cfg)
    assert caught.value.context == {'gene': gene, 'expected': len(setup.reference.gene(gene)), 'actual': 9}


@pytest.mark.parametrize('input_space', ['nucleotide', 'protein'])
def test_protein_class_binding_requires_explicit_label_identity(input_space):
    setup = _setup(input_space, 'protein_classes')
    with pytest.raises(ConfigurationError, match='ordered class labels') as caught:
        validate_reference_binding(setup.reference, setup.config)
    assert caught.value.context['field'] == 'label_schema'
    validate_reference_binding(setup.reference, setup.config, label_schema=setup.label_schema())
    validate_reference_binding(setup.reference, setup.config, protein_reference=setup.protein_reference)


def test_saved_class_order_is_preserved_instead_of_rebuilt_from_source_allele_order():
    setup = _setup('protein', 'protein_classes')
    schema = deepcopy(setup.label_schema())
    schema['genes']['v'].reverse()
    before = deepcopy(schema)
    validate_reference_binding(setup.reference, setup.config, label_schema=schema)
    assert schema == before
    with pytest.raises(ConfigurationError, match='class order'):
        validate_reference_binding(setup.reference, setup.config, label_schema=schema,
                                   protein_reference=setup.protein_reference)


def test_protein_view_and_schema_can_agree_but_may_not_have_different_class_counts():
    setup = _setup('protein', 'protein_classes')
    schema = setup.label_schema()
    validate_reference_binding(setup.reference, setup.config, label_schema=schema,
                               protein_reference=setup.protein_reference)
    # Three allele groups are valid source memberships but are not the view's two equivalence classes.
    schema['genes']['v'] = [[name] for name in setup.reference.gene('V').names]
    cfg = replace(setup.config, v_allele_count=3)
    with pytest.raises(ConfigurationError, match='class order'):
        validate_reference_binding(setup.reference, cfg, label_schema=schema,
                                   protein_reference=setup.protein_reference)


@pytest.mark.parametrize('change,field', [
    ({'d_allele_count': 2}, 'd_allele_count'), ({'num_chain_types': 2}, 'num_chain_types'),
    ({'vocab_size': 9}, 'vocab_size'), ({'max_seq_length': 0}, 'max_seq_length'),
    ({'has_d': 1}, 'has_d'), ({'v_allele_count': True}, 'v_allele_count'),
])
def test_incompatible_dimensions_raise_configuration_errors(change, field):
    setup = _setup(has_d=False)
    with pytest.raises(ConfigurationError) as caught:
        validate_reference_binding(setup.reference, replace(setup.config, **change))
    assert caught.value.context['field'] == field


@pytest.mark.parametrize('reference,config', [(None, None), ('reference', None)])
def test_invalid_binding_objects_raise_domain_errors(reference, config):
    with pytest.raises(ConfigurationError, match='ReferenceSet and AlignAIRConfig'):
        validate_reference_binding(reference, config)


def test_reference_index_contract_is_checked_without_allocating_weights():
    setup = _setup()
    setup.reference.gene('V').index['V1'] = 99
    with pytest.raises(DataContractError, match='order'):
        validate_reference_binding(setup.reference, setup.config)


def test_a_d_head_requires_d_availability_in_the_source():
    setup = _setup()
    setup.reference.has_d = False
    with pytest.raises(ConfigurationError) as caught:
        validate_reference_binding(setup.reference, setup.config)
    assert caught.value.context['field'] == 'has_d'


def test_setup_policy_cannot_disagree_with_its_configuration():
    setup = _setup()
    with pytest.raises(ConfigurationError, match='setup input/label spaces'):
        ModelSetup(setup.reference, setup.config, 'protein', 'alleles')


def test_empty_class_schema_cannot_enable_a_zero_width_head():
    setup = _setup('protein', 'protein_classes')
    schema = setup.label_schema()
    schema['genes']['v'] = []
    with pytest.raises(ConfigurationError) as caught:
        validate_reference_binding(setup.reference, replace(setup.config, v_allele_count=0), label_schema=schema)
    assert caught.value.context['field'] == 'v_label_count'
