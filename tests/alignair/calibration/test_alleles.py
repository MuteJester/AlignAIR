"""Per-head fitting keeps source order, raw truth support and model execution semantics."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from alignair.calibration import calibrate_allele_temperatures
from alignair.core import AlignAIRConfig
from alignair.errors import ConfigurationError, InferenceError, InputError

pytestmark = pytest.mark.integration


class Model(torch.nn.Module):
    def __init__(self, input_space='nucleotide', width=2):
        super().__init__()
        self.cfg = AlignAIRConfig(2, 2, has_d=False, max_seq_length=4, input_space=input_space,
                                   vocab_size=23 if input_space == 'protein' else 6)
        self.child = torch.nn.Dropout().eval()
        self.tokens = []
        self.width = width

    def forward(self, inputs):
        assert not self.training and not torch.is_grad_enabled()
        tokens = inputs['tokenized_sequence']
        self.tokens.extend(tokens.tolist())
        scores = torch.tensor([[.8, .7]])[:, :self.width].repeat(len(tokens), 1)
        return {'v_allele': scores, 'j_allele': scores}


def reference(names=None):
    return SimpleNamespace(gene=lambda gene: SimpleNamespace(names=names or [gene + '0', gene + '1']))


def records(count=51):
    return [{'sequence': ' ac?pT ', 'sequence_aa': 'ACX*', 'v_call': 'V0, V1', 'j_call': 'J1'}
            for _ in range(count)]


@pytest.mark.parametrize('input_space,tokens', [('nucleotide', [1, 4, 5, 5]), ('protein', [1, 2, 21, 22])])
def test_either_alphabet_fits_sigmoid_heads_without_changing_records_or_model_modes(input_space, tokens):
    model, cohort = Model(input_space), records()
    cohort[-1]['j_call'] = None
    original = deepcopy(cohort)
    temperatures, report = calibrate_allele_temperatures(model, reference(), iter(cohort), batch_size=16)
    assert temperatures == {'v': .2, 'j': 3.}
    assert report['v'] == {'T': .2, 'ece_before': .2, 'ece_after': .001, 'acc': 1., 'n': 51}
    assert report['j']['acc'] == 0. and report['j']['n'] == 50
    assert model.tokens == [tokens] * 51
    assert model.training and not model.child.training
    assert model.cfg.allele_temperatures is None and cohort == original


def test_empty_cohort_and_unobserved_or_insufficient_truth_return_empty_reports():
    model = Model()
    assert calibrate_allele_temperatures(model, reference(), []) == ({}, {})
    assert model.tokens == []
    assert calibrate_allele_temperatures(model, reference(), records(1)) == ({}, {})
    cohort = records(3)
    for row in cohort:
        row['v_call'] = ' , '
    temperatures, report = calibrate_allele_temperatures(model, reference(), cohort, genes=('v',), min_samples=1)
    assert temperatures == report == {}


@pytest.mark.parametrize('genes', ['v', (), ('v', 'v'), ('d',), ('missing',)])
def test_invalid_gene_selection_does_not_consume_the_source(genes):
    def untouched():
        pytest.fail('source was consumed for an invalid request')
        yield
    with pytest.raises(ConfigurationError):
        calibrate_allele_temperatures(Model(), reference(), untouched(), genes=genes)


@pytest.mark.parametrize('fields', [{'batch_size': 0}, {'min_samples': 0}])
def test_invalid_counts_fail_before_inference(fields):
    with pytest.raises(ConfigurationError):
        calibrate_allele_temperatures(Model(), reference(), records(), **fields)


def test_categorical_or_unsupported_model_profiles_are_rejected():
    model = Model()
    model.cfg.classification_mode = 'categorical'
    with pytest.raises(ConfigurationError):
        calibrate_allele_temperatures(model, reference(), records())
    model.cfg.classification_mode = 'multilabel'
    model.cfg.input_space = 'other'
    with pytest.raises(ConfigurationError):
        calibrate_allele_temperatures(model, reference(), records())
    with pytest.raises(ConfigurationError):
        calibrate_allele_temperatures(None, reference(), records())


@pytest.mark.parametrize('ref', [None, reference(['V0']), reference(['V0', 'V0']), reference([None, 'V1'])])
def test_reference_must_bind_valid_names_to_the_configured_head(ref):
    with pytest.raises(ConfigurationError):
        calibrate_allele_temperatures(Model(), ref, records(), genes=('v',))


@pytest.mark.parametrize('cohort', [None, 'AC', {}, [None], [{'sequence': 'AC', 'v_call': ['V0']}]])
def test_malformed_records_fail_with_input_errors(cohort):
    model = Model()
    with pytest.raises(InputError):
        calibrate_allele_temperatures(model, reference(), cohort)
    assert not model.tokens


def test_bad_late_protein_input_fails_before_any_model_forward():
    model, cohort = Model('protein'), records(2)
    cohort[-1]['sequence_aa'] = 'AZ'
    with pytest.raises(InputError):
        calibrate_allele_temperatures(model, reference(), cohort, batch_size=1)
    assert not model.tokens


def test_output_width_failure_restores_model_modes():
    model = Model(width=1)
    with pytest.raises(InferenceError, match='width'):
        calibrate_allele_temperatures(model, reference(), records(1))
    assert model.training and not model.child.training


@pytest.mark.parametrize('input_space', ['nucleotide', 'protein'])
def test_real_shared_models_support_both_input_alphabets(input_space):
    import GenAIRR as ga
    from alignair.core import prepare_model
    from alignair.reference import ReferenceSet
    setup = prepare_model(ReferenceSet.from_dataconfigs(ga.HUMAN_IGH_OGRDB), input_space=input_space,
                          has_d=False, max_seq_length=128, embed_dim=4, filters=4, block_out=8)
    model = setup.build_model()
    model.branches['v'].eval()
    modes = [module.training for module in model.modules()]
    cohort = [{'sequence': 'ACGTACGT', 'sequence_aa': 'ACDEFG',
               'v_call': setup.reference.gene('V').names[0]}] * 2
    temperatures, report = calibrate_allele_temperatures(
        model, setup.reference, cohort, genes=('v',), min_samples=2, batch_size=1)
    assert temperatures['v'] > 0 and np.isfinite(temperatures['v'])
    assert report['v']['n'] == 2
    assert [module.training for module in model.modules()] == modes
