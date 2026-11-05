"""Input policy, label identity, and allocation boundaries of shared model preparation."""
from dataclasses import replace

import pytest
import torch

from alignair.core import AlignAIR, AlignAIRConfig, ModelSetup, build_model_config, prepare_model
from alignair.errors import ConfigurationError, DataContractError
from alignair.reference import GeneReference, ReferenceSet
from alignair.reference.reference_set import LocusInfo

pytestmark = pytest.mark.component


def _reference(has_d=True):
    names = {'V': ['V2', 'V1', 'V3'], 'J': ['J2', 'J1'], 'D': ['D2', 'D1']}
    sequences = {'V': ['ATGTGTAAA', 'ATGTGCAAA', 'ATGTGTTTT'], 'J': ['TGGAAA', 'TGGTTT'], 'D': ['TAC', 'ACG']}
    return ReferenceSet({gene: GeneReference(order, sequences[gene], {name: i for i, name in enumerate(order)},
        {name: 3 if gene == 'V' else 0 for name in order} if gene != 'D' else None)
        for gene, order in names.items() if gene != 'D' or has_d}, has_d)


def test_established_defaults_and_reference_derived_label_sizes():
    ref = _reference()
    setup = prepare_model(ref)
    cfg = setup.config
    assert (cfg.architecture, cfg.vocab_size, cfg.use_orientation, cfg.bank_orientation, cfg.anchor_heads,
            cfg.d_presence_head, cfg.reject_empty_sequences, cfg.max_seq_length, cfg.trunk_channels) == (
        'chip-v1', 6, True, True, True, True, False, 576, 64)
    assert setup.reference is ref and cfg.d_allele_count == 2
    assert setup.label_counts == {'v': 3, 'd': 2, 'j': 2}
    assert ref.gene('V').names == ['V2', 'V1', 'V3'] and ref.gene('D').index == {'D2': 0, 'D1': 1}


def test_the_model_is_built_with_the_reference_output_vocabulary():
    setup = prepare_model(_reference(), max_seq_length=192, embed_dim=4, trunk_channels=8)
    model = setup.build_model().eval()
    assert type(model) is AlignAIR
    tokens = torch.zeros(2, 192, dtype=torch.long)
    tokens[:, :12] = 1
    with torch.no_grad():
        output = model({'tokenized_sequence': tokens})
    for gene, size in setup.label_counts.items():
        assert output[setup.config.classification_key(gene)].shape == (2, size)
    assert 'embedding' in model.component_modules()


def test_d_can_be_omitted_without_reordering_or_mutating_the_source():
    ref = _reference()
    before = ref.to_serializable()
    setup = prepare_model(ref, has_d=False)
    assert not setup.config.has_d and setup.config.d_allele_count == 0
    assert 'd' not in setup.label_counts
    assert ref.to_serializable() == before
    assert not prepare_model(_reference(False)).config.has_d


def test_preparation_does_not_allocate_model_parameters(monkeypatch):
    import alignair.core.setup as module
    def forbidden(*args):
        raise AssertionError('model allocated before explicit build_model call')
    monkeypatch.setattr(module, 'AlignAIR', forbidden)
    prepared = prepare_model(_reference())
    assert prepared.config.v_allele_count == 3
    with pytest.raises(AssertionError, match='before explicit'):
        prepared.build_model()


def test_head_choices_remain_independent_of_the_input_and_label_identity():
    setup = prepare_model(_reference(), use_orientation=False, bank_orientation=False, anchor_heads=False,
                          d_presence_head=False)
    assert setup.label_space == 'alleles' and setup.config.v_allele_count == 3
    assert not (setup.config.use_orientation or setup.config.anchor_heads or setup.config.d_presence_head)


def test_locus_head_count_comes_from_the_ordered_reference_schema():
    ref = _reference()
    ref.loci = [LocusInfo('IGH', 'BCR_HEAVY', True, {}), LocusInfo('TRB', 'TCR_BETA', True, {})]
    assert prepare_model(ref).config.num_chain_types == 2


@pytest.mark.parametrize('options,field', [
    ({'input_space': 'aa'}, 'input_space'), ({'label_space': 'classes'}, 'label_space'),
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


@pytest.mark.parametrize('counts,options', [
    ((0, 2), {}), ((True, 2), {}), ((2, -1), {}), ((2, 2, 0), {'has_d': True}),
    ((2, 2), {'num_chain_types': 0}), ((2, 2), {'has_d': None}),
])
def test_count_only_factory_rejects_invalid_dimensions(counts, options):
    with pytest.raises(ConfigurationError):
        build_model_config(*counts, **options)


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


def test_mutated_config_is_checked_again_before_allocation():
    setup = prepare_model(_reference())
    setup.config.v_allele_count = 99
    with pytest.raises(ConfigurationError, match='head size'):
        setup.build_model()


@pytest.mark.parametrize('config', [lambda: None, lambda: AlignAIRConfig(3, 2, has_d=False, d_allele_count=2),
    lambda: AlignAIRConfig(3, 2, has_d=False, num_chain_types=2), lambda: AlignAIRConfig(3, 2, has_d='yes')])
def test_hand_built_setup_cannot_bypass_binding_checks(config):
    with pytest.raises(ConfigurationError):
        ModelSetup(_reference(), config(), 'nucleotide', 'alleles')


def test_reference_based_preparation_does_not_require_the_simulator(monkeypatch):
    import builtins
    original = builtins.__import__
    def without_simulator(name, *args, **kwargs):
        if name == 'GenAIRR' or name.startswith('GenAIRR.'):
            raise AssertionError('core preparation imported the simulator')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_simulator)
    prepared = prepare_model(_reference())
    assert prepared.label_counts == {'v': 3, 'd': 2, 'j': 2}
