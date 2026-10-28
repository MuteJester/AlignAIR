"""Categorical probabilities: explicit head selection, one forward and tokenizer binding."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from alignair.data import SequenceInputAdapter
from alignair.data.tokenizer import AAProteinTokenizer
from alignair.errors import InferenceError, InputError, NumericalError, current_error_context
from alignair.predict import class_probabilities
from alignair.predict.probabilities import class_probs, protein_class_probabilities, softmax_probabilities

pytestmark = pytest.mark.component


class Heads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.dropout = torch.nn.Dropout(.9)
    def forward(self, inputs):
        self.calls += 1
        value = self.dropout(inputs['tokenized_sequence'][:,0].float())
        return {'v':torch.stack((value,value+np.log(3.)),dim=1),
                'j':torch.stack((value,value),dim=1)}


def test_multiple_heads_use_one_forward_per_batch_and_have_hand_calculated_probabilities():
    model = Heads()
    adapter = SequenceInputAdapter(AAProteinTokenizer(),4)
    output = class_probabilities(model,['AC','DE','FG'],adapter,keys=('v','j'),batch_size=2)
    assert model.calls == 2
    assert np.allclose(output['v'],[[.25,.75]]*3)
    assert np.array_equal(output['j'],[[.5,.5]]*3)
    assert model.training and model.dropout.training
    repeated = class_probabilities(model,['AC','DE','FG'],adapter,keys=('v','j'),batch_size=2)
    assert all(np.array_equal(output[key],repeated[key]) for key in output)


@pytest.mark.parametrize('bad', [None,torch.ones(2),torch.ones(1,0),torch.ones(1,2,dtype=torch.long)])
def test_softmax_requires_categorical_logit_matrices(bad):
    with pytest.raises(InferenceError): softmax_probabilities(bad)


def test_softmax_rejects_nonfinite_logits_and_supports_explicit_placement_and_bfloat16():
    with pytest.raises(NumericalError): softmax_probabilities(torch.tensor([[float('nan')]]))
    result = softmax_probabilities(torch.zeros(2,3,dtype=torch.bfloat16),device='cpu')
    assert result.dtype == np.float32 and np.allclose(result,1/3)


def test_empty_batches_and_unspecified_head_semantics_are_rejected():
    adapter = SequenceInputAdapter(AAProteinTokenizer(),4)
    with pytest.raises(InputError): class_probabilities(Heads(),[],adapter,keys=('v',))
    with pytest.raises(InputError): class_probabilities(Heads(),['AC'],adapter,keys=None)


@pytest.mark.parametrize('sequences', [[],[''],['ac'],['AZ'],['AAAAA'],['AC','AZ']])
def test_research_protein_policy_is_strict_and_rejects_bad_late_rows_before_inference(sequences):
    model = Heads()
    with pytest.raises(InputError):
        protein_class_probabilities(model,sequences,SimpleNamespace(max_seq_length=4),'cpu',('v',),batch=1)
    assert model.calls == 0


def test_single_head_research_probabilities_accept_unknown_and_stop_residues():
    result = class_probs(Heads(),['X*'],SimpleNamespace(max_seq_length=4),'cpu','v')
    assert result.shape == (1,2)


def test_selected_probability_heads_ignore_changing_unselected_outputs():
    model = Heads()
    original = model.forward
    def forward(inputs):
        output = original(inputs)
        if model.calls == 1:
            output['unused'] = torch.full((len(inputs['tokenized_sequence']),), float('nan'))
        return output
    model.forward = forward
    result = class_probabilities(model, ['AC', 'DE', 'FG'],
                                SequenceInputAdapter(AAProteinTokenizer(), 4), keys=('v',), batch_size=2)
    assert list(result) == ['v'] and np.allclose(result['v'], [[.25, .75]] * 3)


def test_probability_width_changes_have_head_and_source_offset():
    model = Heads()
    original = model.forward
    def forward(inputs):
        output = original(inputs)
        if model.calls > 1:
            output['v'] = output['v'][:, :1]
        return output
    model.forward = forward
    with pytest.raises(InferenceError, match='shape or dtype') as caught:
        class_probabilities(model, ['AC', 'DE', 'FG'], SequenceInputAdapter(AAProteinTokenizer(), 4),
                            keys=('v',), batch_size=2)
    assert caught.value.context['record_offset'] == 2 and caught.value.context['field'] == 'v'
    assert model.training and model.dropout.training and torch.is_grad_enabled()
    assert current_error_context() == {}


@pytest.mark.parametrize('device', ['cpu', None])
def test_collection_preserves_per_batch_softmax_and_explicit_device(monkeypatch, device):
    import alignair.predict.probabilities as probabilities
    seen = []
    original = probabilities.softmax_probabilities
    def observe(logits, *, device):
        seen.append((len(logits), device))
        return original(logits, device=device)
    monkeypatch.setattr(probabilities, 'softmax_probabilities', observe)
    result = class_probabilities(Heads(), ['AC', 'DE', 'FG'], SequenceInputAdapter(AAProteinTokenizer(), 4),
                                keys=('v', 'j'), batch_size=2, probability_device=device)
    assert seen == [(2, device), (2, device), (1, device), (1, device)]
    assert np.allclose(result['v'], [[.25, .75]] * 3)
