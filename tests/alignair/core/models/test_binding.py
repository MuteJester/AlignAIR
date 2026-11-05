"""One preflight contract for prepared, saved and restored reference-bound configurations."""
from dataclasses import replace

import pytest
import torch

from alignair.core import AlignAIR, ModelSetup, prepare_model, validate_reference_binding
from alignair.core.setup import setup_from_schema
from alignair.errors import ConfigurationError, DataContractError, ModelIntegrityError
from alignair.reference import ReferenceSet

pytestmark = pytest.mark.component


def _setup(has_d=True):
    reference = ReferenceSet.from_genotype({
        'V': {'V2': 'ATGTGTAAA', 'V1': 'ATGTGCAAA', 'V3': 'ATGTGTTTT'},
        'D': {'D1': 'TAC'}, 'J': {'J2': 'TGGAAA', 'J1': 'TGGTTT'},
    })
    reference.gene('V').anchors = {name: 3 for name in reference.gene('V').names}
    reference.gene('J').anchors = {name: 0 for name in reference.gene('J').names}
    return prepare_model(reference, has_d=has_d,
                         max_seq_length=128, embed_dim=4, filters=4, block_out=8)


@pytest.mark.parametrize('has_d', [False, True])
def test_allele_binding_needs_no_schema_and_allows_an_omitted_d_head(has_d):
    setup = _setup(has_d=has_d)
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
    with pytest.raises(ConfigurationError, match='nucleotide'):
        ModelSetup(setup.reference, setup.config, 'protein', 'alleles')


def test_prepared_restoration_preserves_weights_and_predictions():
    setup = _setup()
    original = setup.build_model().eval()
    restored = setup.restore_model(original.state_dict(), device='cpu')
    assert restored is not original and restored.cfg == setup.config and restored.cfg is not setup.config
    assert all(not module.training for module in restored.modules())
    tokens = torch.ones(2, 128, dtype=torch.long)
    with torch.no_grad():
        expected, actual = original({'tokenized_sequence': tokens}), restored({'tokenized_sequence': tokens})
    assert expected.keys() == actual.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    for key, weights in original.state_dict().items():
        assert torch.equal(restored.state_dict()[key], weights)
        assert restored.state_dict()[key].data_ptr() != weights.data_ptr()


def test_prepared_restoration_revalidates_before_allocation(tmp_path, monkeypatch):
    setup = _setup()
    setup.config.v_allele_count = 99
    def forbidden(*args, **kwargs):
        raise AssertionError('restoration allocated before binding preflight')
    monkeypatch.setattr(AlignAIR, '__init__', forbidden)
    source = tmp_path / 'candidate.alignair'
    with pytest.raises(ConfigurationError, match='head size') as caught:
        setup.restore_model({}, source=source)
    assert caught.value.context['stage'] == 'model_binding'
    assert caught.value.context['path'] == str(source)


def test_prepared_restoration_retains_weight_failure_diagnostics(tmp_path):
    setup = _setup()
    state = setup.build_model().state_dict()
    del state[next(iter(state))]
    source = tmp_path / 'candidate.alignair'
    with pytest.raises(ModelIntegrityError, match='Missing key') as caught:
        setup.restore_model(state, source=source)
    assert caught.value.context['stage'] == 'model_restoration'
    assert caught.value.context['path'] == str(source)
    assert isinstance(caught.value.__cause__, RuntimeError)


@pytest.mark.parametrize('problem', ['missing_gene', 'wrong_width', 'allele_order'])
def test_schema_preflight_happens_before_model_allocation(problem, monkeypatch):
    setup = _setup()
    schema = setup.label_schema()
    if problem == 'missing_gene':
        del schema['genes']['v']
    elif problem == 'wrong_width':
        schema['genes']['v'].pop()
    else:
        schema['genes']['v'].reverse()
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid saved labels reached model allocation')
    monkeypatch.setattr(AlignAIR, '__init__', forbidden)
    with pytest.raises(ConfigurationError):
        setup_from_schema(setup.reference, setup.config, schema)
