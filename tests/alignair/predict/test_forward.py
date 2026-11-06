"""One inference primitive: detached outputs, checked shapes, lazy batches and mode restoration."""
import numpy as np
import pytest
import torch

from alignair.core import evaluation_mode
from alignair.errors import ConfigurationError, InferenceError, InputError, NumericalError, current_error_context
from alignair.predict import PredictConfig, collect_prediction_outputs, forward_batch, prediction_batches
from alignair.predict.forward import PredictionBatch, output_numpy, run_model

pytestmark = pytest.mark.component


class Echo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = torch.nn.Dropout().eval()
        self.calls = 0
    def forward(self, inputs):
        assert not self.training and not torch.is_grad_enabled()
        self.calls += 1
        return inputs


def test_selected_outputs_are_detached_and_mixed_modes_restore_before_return():
    model = Echo()
    source = torch.ones(2,3,requires_grad=True)
    result = forward_batch(model,{'scores':source,'unused':torch.full((2,),float('nan'))},
                           expected_rows=2,output_keys=('scores',))
    assert set(result) == {'scores'} and not result['scores'].requires_grad
    assert model.training and not model.child.training
    assert current_error_context() == {}


@pytest.mark.parametrize('outputs', [None,{}, {'x':None}, {'x':torch.tensor(1.)},
                                    {'x':torch.ones(2)}, {'x':torch.ones(1,dtype=torch.complex64)},
                                    {1:torch.ones(1)}, {'':torch.ones(1)}])
def test_bad_outputs_raise_inference_errors_and_restore_modes(outputs):
    model = Echo()
    model.forward = lambda _: outputs
    with pytest.raises(InferenceError):
        forward_batch(model,{},expected_rows=1)
    assert model.training and not model.child.training
    assert current_error_context() == {}


def test_nonfinite_requested_output_has_field_context():
    with pytest.raises(NumericalError) as caught:
        forward_batch(Echo(),{'bad':torch.tensor([float('inf')])},expected_rows=1)
    assert caught.value.context == {'stage':'model_inference','field':'bad'}


@pytest.mark.parametrize('keys', [(), 'x', ('x','x'), (1,), ('',), 1])
def test_invalid_output_selection_fails_before_forward(keys):
    model = Echo()
    with pytest.raises(ConfigurationError):
        forward_batch(model,{},expected_rows=1,output_keys=keys)
    assert model.calls == 0


def test_invalid_assembly_and_missing_keys_are_classified():
    for model, inputs in [(None,{}),(Echo(),None)]:
        with pytest.raises(ConfigurationError):
            forward_batch(model,inputs,expected_rows=1)
    with pytest.raises(InferenceError):
        forward_batch(Echo(),{'other':torch.ones(1)},expected_rows=1,output_keys=('missing',))


@pytest.mark.parametrize('count', [True,False,0,-1,1.5,None])
def test_invalid_row_counts_are_configuration_errors(count):
    with pytest.raises(ConfigurationError):
        forward_batch(Echo(),{},expected_rows=count)


def test_batches_are_lazy_include_tail_and_end_all_execution_scopes_before_yield():
    model, calls = Echo(), []
    def source():
        calls.append('opened')
        yield from range(5)
    def adapter(rows):
        calls.append(tuple(rows))
        return {'values':torch.tensor(rows)}
    batches = prediction_batches(model,source(),adapter,batch_size=2)
    assert calls == [] and model.calls == 0
    first = next(batches)
    assert first.record_offset == 0 and first.size == 2
    assert model.training and not model.child.training and torch.is_grad_enabled()
    assert current_error_context() == {}
    remaining = list(batches)
    assert [(batch.record_offset,batch.size) for batch in remaining] == [(2,2),(4,1)]
    assert calls == ['opened',(0,1),(2,3),(4,)]


def test_empty_batches_do_not_adapt_or_forward_and_invalid_setup_is_eager():
    model = Echo()
    assert list(prediction_batches(model,[],lambda _:pytest.fail('adapted'),batch_size=2)) == []
    for value in (0,True):
        with pytest.raises(ConfigurationError):
            prediction_batches(model,[],lambda _: {},batch_size=value)
    for model,adapter in [(None,lambda _: {}),(Echo(),None)]:
        with pytest.raises(ConfigurationError):
            prediction_batches(model,[],adapter,batch_size=2)


def test_injected_forward_failure_restores_modes():
    model = Echo()
    def fail(_): raise RuntimeError('injected')
    model.forward = fail
    with pytest.raises(RuntimeError,match='injected'):
        forward_batch(model,{},expected_rows=1)
    assert model.training and not model.child.training


def test_numpy_snapshots_are_independent_and_bfloat16_is_representable():
    source = torch.tensor([1.])
    copied = output_numpy(source)
    source.fill_(9)
    assert copied.tolist() == [1.]
    assert output_numpy(torch.ones(2,dtype=torch.bfloat16)).dtype == np.float32


def test_nucleotide_forward_uses_shared_normalization_unknowns_and_cropping():
    model = Echo()
    batches = run_model(model,['acgtn?','a?p','nn'],PredictConfig(max_seq_length=4,batch_size=2))
    # the window is 3 bases: the model's width less the terminal end slot
    assert batches[0]['tokenized_sequence'].tolist() == [[1,4,3,0],[1,5,5,0]]
    assert batches[1]['tokenized_sequence'].tolist() == [[5,5,0,0]]
    assert model.training


def test_historical_evaluation_context_is_the_shared_core_runtime_owner():
    from alignair.train.engine import evaluation_mode as old
    assert old is evaluation_mode


def test_collection_snapshots_reused_storage_and_keeps_tail_and_first_key_order():
    buffer = torch.zeros(2, 3)
    def batches():
        yield PredictionBatch(0, 2, {'scores': buffer, 'positions': torch.tensor([4, 5])})
        buffer.fill_(7)
        yield PredictionBatch(2, 1, {'positions': torch.tensor([6]), 'scores': buffer[:1]})
    result = collect_prediction_outputs(batches())
    assert list(result) == ['scores', 'positions']
    assert result['scores'].tolist() == [[0, 0, 0], [0, 0, 0], [7, 7, 7]]
    assert result['positions'].tolist() == [4, 5, 6]


@pytest.mark.parametrize('outputs', [{}, {'other': torch.ones(1, 3)},
                                   {'scores': torch.ones(1, 3), 'extra': torch.ones(1)}])
def test_collection_rejects_changed_heads_with_batch_context(outputs):
    batches = [PredictionBatch(0, 2, {'scores': torch.ones(2, 3)}), PredictionBatch(2, 1, outputs)]
    with pytest.raises(InferenceError, match='keys') as caught:
        collect_prediction_outputs(batches)
    assert caught.value.context['record_offset'] == 2
    assert caught.value.context['batch_size'] == 1
    assert caught.value.context['expected_keys'] == ['scores']
    assert current_error_context() == {}


@pytest.mark.parametrize('later', [torch.ones(1, 4), torch.ones(1, 3, 1),
                                 torch.ones(1, 3, dtype=torch.float64)])
def test_collection_rejects_changed_width_rank_or_dtype_and_restores_runtime(later):
    model = Echo()
    def adapter(rows):
        return {'scores': torch.ones(2, 3) if len(rows) == 2 else later}
    with pytest.raises(InferenceError, match='shape or dtype') as caught:
        collect_prediction_outputs(prediction_batches(model, range(3), adapter, batch_size=2))
    assert caught.value.context['record_offset'] == 2 and caught.value.context['field'] == 'scores'
    assert model.training and not model.child.training and torch.is_grad_enabled()
    assert current_error_context() == {}


@pytest.mark.parametrize('transformed', [None, np.array(1.), np.ones((1, 3))])
def test_collection_checks_transform_rows_and_array_contract(transformed):
    with pytest.raises(InferenceError, match='one row per input') as caught:
        collect_prediction_outputs([PredictionBatch(0, 2, {'scores': torch.ones(2, 3)})],
                                   transform=lambda _: transformed)
    assert caught.value.context['expected_rows'] == 2 and caught.value.context['field'] == 'scores'


def test_collection_snapshots_transform_buffers_before_advancing():
    buffer = np.zeros((1, 2))
    def batches():
        yield PredictionBatch(0, 1, {'scores': torch.ones(1, 2)})
        buffer.fill(9)
        yield PredictionBatch(1, 1, {'scores': torch.ones(1, 2)})
    result = collect_prediction_outputs(batches(), transform=lambda _: buffer)
    assert result['scores'].tolist() == [[0, 0], [9, 9]]


def test_empty_collection_has_explicit_input_error():
    with pytest.raises(InputError, match='nonempty'):
        collect_prediction_outputs([])
