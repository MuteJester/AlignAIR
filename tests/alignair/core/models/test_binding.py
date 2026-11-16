"""One preflight contract for prepared, saved and restored reference-bound configurations."""
from dataclasses import replace

import pytest
import torch

from alignair.core import prepare_model, validate_reference_binding
from alignair.errors import ConfigurationError, DataContractError, ModelIntegrityError
from alignair.reference import ReferenceSet

pytestmark = pytest.mark.component


def _setup(has_d=True):
    reference = ReferenceSet.from_alleles({
        'V': {'V2': 'ATGTGTAAA', 'V1': 'ATGTGCAAA', 'V3': 'ATGTGTTTT'},
        'D': {'D1': 'TAC'}, 'J': {'J2': 'TGGAAA', 'J1': 'TGGTTT'},
    })
    reference.gene('V').anchors = {name: 3 for name in reference.gene('V').names}
    reference.gene('J').anchors = {name: 0 for name in reference.gene('J').names}
    return prepare_model(reference, has_d=has_d,
                         max_seq_length=128, embed_dim=4, trunk_channels=8)


@pytest.mark.parametrize('has_d', [False, True])
def test_a_binding_allows_an_omitted_d_head(has_d):
    setup = _setup(has_d=has_d)
    before = setup.reference.to_serializable()
    validate_reference_binding(setup.reference, setup.config)
    assert setup.reference.to_serializable() == before


@pytest.mark.parametrize('change,field', [
    ({'num_chain_types': 2}, 'num_chain_types'), ({'vocab_size': 9}, 'vocab_size'),
    ({'max_seq_length': 0}, 'max_seq_length'), ({'has_d': 1}, 'has_d'),
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


@pytest.mark.parametrize('gene', ['V', 'J'])
def test_a_reference_without_alleles_for_a_gene_cannot_be_bound(gene):
    setup = _setup()
    with pytest.raises(ConfigurationError, match=f'no {gene} alleles'):
        validate_reference_binding(setup.reference.subset({gene: []}), setup.config)


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
