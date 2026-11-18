"""The configuration a reference gives a model, and the allocation boundary of shared model preparation."""
import pytest
import torch

from alignair.core import AlignAIR, AlignAIRConfig, ModelSetup, prepare_model
from alignair.errors import ConfigurationError, DataContractError
from alignair.reference import GeneReference, ReferenceSet
from alignair.reference.loci import LocusInfo

pytestmark = pytest.mark.component


def _reference(has_d=True):
    names = {'V': ['V2', 'V1', 'V3'], 'J': ['J2', 'J1'], 'D': ['D2', 'D1']}
    sequences = {'V': ['ATGTGTAAA', 'ATGTGCAAA', 'ATGTGTTTT'], 'J': ['TGGAAA', 'TGGTTT'], 'D': ['TAC', 'ACG']}
    return ReferenceSet({gene: GeneReference(order, sequences[gene], {name: i for i, name in enumerate(order)},
        {name: 3 if gene == 'V' else 0 for name in order} if gene != 'D' else None)
        for gene, order in names.items() if gene != 'D' or has_d}, has_d)


def _allele_counts(setup):
    return {gene: len(setup.reference.gene(gene.upper())) for gene in setup.config.genes}


def test_established_defaults_and_reference_derived_label_sizes():
    ref = _reference()
    setup = prepare_model(ref)
    cfg = setup.config
    assert (cfg.architecture, cfg.vocab_size, cfg.bank_orientation, cfg.max_seq_length, cfg.trunk_channels) == (
        'chip-v1', 6, True, 576, 64)
    assert setup.reference is ref and cfg.has_d
    assert _allele_counts(setup) == {'v': 3, 'd': 2, 'j': 2}
    assert ref.gene('V').names == ['V2', 'V1', 'V3'] and ref.gene('D').index == {'D2': 0, 'D1': 1}


def test_the_model_is_built_with_the_reference_output_vocabulary():
    setup = prepare_model(_reference(), max_seq_length=192, embed_dim=4, trunk_channels=8)
    model = setup.build_model().eval()
    assert type(model) is AlignAIR
    tokens = torch.zeros(2, 192, dtype=torch.long)
    tokens[:, :12] = 1
    with torch.no_grad():
        output = model({'tokenized_sequence': tokens})
    for gene, size in _allele_counts(setup).items():
        assert output[f"{gene}_allele"].shape == (2, size)
    assert 'embedding' in model.component_modules()


def test_d_can_be_omitted_without_reordering_or_mutating_the_source():
    ref = _reference()
    before = ref.to_serializable()
    setup = prepare_model(ref, has_d=False)
    assert not setup.config.has_d and 'd' not in setup.config.genes
    assert ref.to_serializable() == before
    assert not prepare_model(_reference(False)).config.has_d


def test_preparation_does_not_allocate_model_parameters(monkeypatch):
    import alignair.core.setup as module
    def forbidden(*args):
        raise AssertionError('model allocated before explicit build_model call')
    monkeypatch.setattr(module, 'AlignAIR', forbidden)
    prepared = prepare_model(_reference())
    assert prepared.config.genes == ('v', 'd', 'j')
    with pytest.raises(AssertionError, match='before explicit'):
        prepared.build_model()


def test_the_bank_orientation_choice_is_independent_of_the_reference():
    setup = prepare_model(_reference(), bank_orientation=False)
    assert not setup.config.bank_orientation and _allele_counts(setup) == {'v': 3, 'd': 2, 'j': 2}


def test_locus_head_count_comes_from_the_ordered_reference_schema():
    ref = _reference()
    ref.loci = [LocusInfo('IGH', 'BCR_HEAVY', True, {}), LocusInfo('TRB', 'TCR_BETA', True, {})]
    assert prepare_model(ref).config.num_chain_types == 2


@pytest.mark.parametrize('options,field', [
    ({'has_d': 1}, 'has_d'), ({'max_seq_length': True}, 'max_seq_length'),
    ({'max_seq_length': .5}, 'max_seq_length'), ({'max_seq_length': 0}, 'max_seq_length'),
    ({'vocab_size': 7}, 'vocab_size'),
])
def test_invalid_policies_raise_domain_errors(options, field):
    with pytest.raises(ConfigurationError) as caught:
        prepare_model(_reference(), **options)
    assert caught.value.context['field'] == field


@pytest.mark.parametrize('options', [
    {'v_allele_count': 9}, {'v_label_count': 9},
    {'num_chain_types': 5}, {'missing_setting': True},
])
def test_incompatible_binding_and_unknown_options_are_rejected(options):
    with pytest.raises(ConfigurationError):
        prepare_model(_reference(), **options)


@pytest.mark.parametrize('options', [
    {'num_chain_types': 0}, {'num_chain_types': True}, {'has_d': None}, {'embed_dim': 1.5}, {'trunk_channels': 0},
    {'bank_orientation': 1}, {'vocab_size': 5}, {'architecture': ''},
])
def test_a_configuration_refuses_invalid_dimensions_when_it_is_built(options):
    with pytest.raises(ConfigurationError):
        AlignAIRConfig(**options)


def test_invalid_reference_type_and_unavailable_d_are_rejected():
    with pytest.raises(ConfigurationError, match='ReferenceSet'):
        prepare_model('HUMAN_IGH_OGRDB')
    with pytest.raises(ConfigurationError, match='D head is unavailable'):
        prepare_model(_reference(False), has_d=True)


def test_bad_allele_order_is_rejected_by_the_shared_reference_validator():
    ref = _reference()
    ref.gene('V').index['V2'] = 2
    with pytest.raises(DataContractError, match='order'):
        prepare_model(ref)
    with pytest.raises(DataContractError, match='order'):
        ref.validate()
    ref = _reference(False)
    ref.validate()
    with pytest.raises(ConfigurationError, match='absent'):
        ref.validate(genes=('D',))


@pytest.mark.parametrize('reference,config', [
    (_reference, lambda: None), (_reference, lambda: AlignAIRConfig(has_d=False, num_chain_types=2)),
    (_reference, lambda: AlignAIRConfig(has_d='yes')), (lambda: _reference(False), lambda: AlignAIRConfig(has_d=True))])
def test_hand_built_setup_cannot_bypass_binding_checks(reference, config):
    with pytest.raises(ConfigurationError):
        ModelSetup(reference(), config())


def test_reference_based_preparation_does_not_require_the_simulator(monkeypatch):
    import builtins
    original = builtins.__import__
    def without_simulator(name, *args, **kwargs):
        if name == 'GenAIRR' or name.startswith('GenAIRR.'):
            raise AssertionError('core preparation imported the simulator')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_simulator)
    prepared = prepare_model(_reference())
    assert _allele_counts(prepared) == {'v': 3, 'd': 2, 'j': 2}
