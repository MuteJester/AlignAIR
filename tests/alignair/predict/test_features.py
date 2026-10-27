"""Component inspection preserves model behavior and cleans hooks on every failure path."""
import numpy as np
import pytest
import torch

from alignair.data import SequenceInputAdapter
from alignair.data.tokenizer import AAProteinTokenizer
from alignair.errors import ConfigurationError, InferenceError, NumericalError
from alignair.predict import component_features, forward_batch

pytestmark = pytest.mark.component


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v = torch.nn.Linear(4,2,bias=False)
        self.j = torch.nn.Linear(4,2,bias=False)
        self.v.eval()
    def component_modules(self):
        return {'v.cls_head':self.v,'j.cls_head':self.j}
    def forward(self, inputs):
        values = inputs['tokenized_sequence'].float()
        return {'v':self.v(values),'j':self.j(values+1)}


def adapter(): return SequenceInputAdapter(AAProteinTokenizer(),4)


def test_feature_order_and_width_match_component_inputs_without_changing_predictions():
    model = Model()
    inputs = adapter()(['AC','DE'])
    before = forward_batch(model,inputs,expected_rows=2)
    result = component_features(model,['AC','DE'],adapter(),components=('j.cls_head','v.cls_head'))
    expected = np.concatenate((inputs['tokenized_sequence'].numpy()+1,inputs['tokenized_sequence'].numpy()),axis=1)
    assert np.array_equal(result,expected)
    after = forward_batch(model,inputs,expected_rows=2)
    assert all(torch.equal(before[key],after[key]) for key in before)
    assert model.training and not model.v.training and model.j.training
    assert not model.v._forward_pre_hooks and not model.j._forward_pre_hooks


@pytest.mark.parametrize('failure', ['missing','alias','registry','components','model','adapter','unnamed'])
def test_bad_capture_requests_fail_before_installing_hooks(failure):
    model = Model()
    components, input_adapter = ('v.cls_head','j.cls_head'), adapter()
    if failure == 'missing': components = ('v.cls_head','missing')
    elif failure == 'alias': model.component_modules = lambda: {'v.cls_head':model.v,'j.cls_head':model.v}
    elif failure == 'registry': model.component_modules = lambda: None
    elif failure == 'components': components = None
    elif failure == 'adapter': input_adapter = None
    elif failure == 'unnamed': model.component_modules = None
    else: model = object()
    with pytest.raises(ConfigurationError):
        component_features(model,['AC'],input_adapter,components=components)
    if isinstance(model,Model):
        assert not model.v._forward_pre_hooks and not model.j._forward_pre_hooks


@pytest.mark.parametrize('failure', ['unused','twice','shape','nonfinite','forward','registration'])
def test_capture_failures_remove_all_hooks_and_restore_modes(failure):
    model = Model()
    error = InferenceError
    if failure == 'unused': model.forward = lambda inputs: {'v':model.v(inputs['tokenized_sequence'].float())}
    elif failure == 'twice':
        def twice(inputs):
            values = inputs['tokenized_sequence'].float()
            return {'v':model.v(values)+model.v(values),'j':model.j(values)}
        model.forward = twice
    elif failure == 'shape':
        model.forward = lambda inputs: {'v':model.v(inputs['tokenized_sequence'][0].float())}
    elif failure == 'nonfinite':
        model.forward = lambda inputs: {'v':model.v(torch.full((1,4),float('nan')))}
        error = NumericalError
    elif failure == 'registration':
        def fail(*a,**k): raise RuntimeError('registration failed')
        model.j.register_forward_pre_hook = fail
        error = RuntimeError
    else:
        def fail(inputs): raise RuntimeError('forward failed')
        model.forward = fail
        error = RuntimeError
    with pytest.raises(error):
        component_features(model,['AC'],adapter(),components=('v.cls_head','j.cls_head'))
    assert model.training and not model.v.training
    assert not model.v._forward_pre_hooks and not model.j._forward_pre_hooks
