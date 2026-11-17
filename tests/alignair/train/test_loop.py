"""Step-window and source-consumption contracts shared by nucleotide and protein runners."""
from itertools import count
from types import SimpleNamespace

import pytest
import torch

from alignair.data.batching import TrainingBatch
from alignair.errors import ConfigurationError, TrainingDataError, TrainingError, current_error_context
from alignair.train.engine import StepResult, TrainingEngine
from alignair.train.loop import training_iterations

pytestmark = pytest.mark.component


class RecordingEngine:
    def __init__(self):
        self.calls = []

    def step(self, batch, *, step):
        self.calls.append((step, batch))
        return StepResult(float(step), {'loss': float(step)}, 0.)


def test_resume_window_uses_absolute_steps_and_drops_only_the_incomplete_tail():
    engine = RecordingEngine()
    adapted = []
    def adapt(rows):
        adapted.append(rows)
        return rows
    results = list(training_iterations(engine, range(7), adapt, start_step=10, stop_step=20, batch_size=3))
    assert [iteration.step for iteration in results] == [11, 12]
    assert [iteration.records for iteration in results] == [(0, 1, 2), (3, 4, 5)]
    assert adapted == [(0, 1, 2), (3, 4, 5)]
    assert [step for step, _ in engine.calls] == [11, 12]


def test_step_budget_and_caller_stop_do_not_prefetch_the_next_batch():
    pulled, adapted = [], []
    def records():
        for value in count():
            pulled.append(value)
            yield value
    def adapt(rows):
        adapted.append(rows)
        return rows
    engine = RecordingEngine()
    iterations = training_iterations(engine, records(), adapt, stop_step=3, batch_size=2)
    assert pulled == [] and adapted == []
    first = next(iterations)
    assert first.step == 1 and pulled == [0, 1]
    # A runner can checkpoint, validate, or stop here without advancing the source or optimizer.
    assert len(engine.calls) == len(adapted) == 1
    assert [iteration.step for iteration in iterations] == [2, 3]
    assert pulled == list(range(6)) and len(engine.calls) == 3


def test_closing_an_iteration_stream_does_not_close_the_callers_source():
    source = iter(range(8))
    iterations = training_iterations(RecordingEngine(), source, tuple, stop_step=5, batch_size=2)
    next(iterations)
    iterations.close()
    assert next(source) == 2


@pytest.mark.parametrize('start,stop', [(0, 0), (7, 7)])
def test_empty_window_never_initializes_a_source_or_adapter(start, stop):
    class Unreadable:
        def __iter__(self):
            raise AssertionError('source was initialized')
    def unexpected(rows):
        raise AssertionError('adapter was called')
    assert list(training_iterations(RecordingEngine(), Unreadable(), unexpected,
                start_step=start, stop_step=stop, batch_size=2)) == []


@pytest.mark.parametrize('values,field', [
    ({'start_step': -1}, 'start_step'), ({'start_step': True}, 'start_step'),
    ({'stop_step': -1}, 'stop_step'), ({'stop_step': 2.5}, 'stop_step'),
    ({'batch_size': 0}, 'batch_size'), ({'batch_size': False}, 'batch_size'),
    ({'batch_size': '2'}, 'batch_size'),
])
def test_invalid_counts_are_rejected_before_source_iteration(values, field):
    options = {'start_step': 0, 'stop_step': 2, 'batch_size': 2, **values}
    with pytest.raises(ConfigurationError) as caught:
        training_iterations(RecordingEngine(), None, tuple, **options)
    assert caught.value.context['field'] == field


def test_reversed_window_and_noncallable_adapter_are_configuration_errors():
    with pytest.raises(ConfigurationError, match='at least start_step'):
        training_iterations(RecordingEngine(), None, tuple, start_step=5, stop_step=4, batch_size=2)
    with pytest.raises(ConfigurationError, match='batch_adapter'):
        training_iterations(RecordingEngine(), None, None, stop_step=2, batch_size=2)


@pytest.mark.parametrize('failure', ['source_init', 'source_read', 'adapter', 'engine'])
def test_failure_reports_the_attempted_step_without_emitting_success_or_leaking_context(failure):
    class Source:
        def __iter__(self):
            if failure == 'source_init':
                raise TrainingDataError('source init failed')
            return self
        def __next__(self):
            if failure == 'source_read':
                raise TrainingDataError('source read failed')
            return 1
    def adapt(rows):
        if failure == 'adapter':
            raise TrainingDataError('batch adaptation failed')
        return rows
    def update(batch, *, step):
        raise TrainingError('optimizer update failed')
    engine = SimpleNamespace(step=update) if failure == 'engine' else RecordingEngine()
    iterations = training_iterations(engine, Source(), adapt, start_step=8, stop_step=10, batch_size=2)
    with pytest.raises((TrainingError, TrainingDataError)) as caught:
        next(iterations)
    assert caught.value.context['step'] == 9
    assert caught.value.context['stage'] == 'training_iteration'
    assert list(iterations) == []
    assert current_error_context() == {}


def test_runner_code_executes_outside_iteration_error_context():
    iteration = next(training_iterations(RecordingEngine(), [1], tuple, stop_step=1, batch_size=1))
    assert iteration.result.total == 1
    assert current_error_context() == {}
    error = TrainingError('runner callback failed')
    assert 'step' not in error.context


def test_real_optimizer_and_scheduler_advance_once_for_each_complete_batch():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))
        def forward(self, inputs):
            return self.weight * inputs['x']
    class Tasks:
        def loss(self, output, targets):
            loss = (output - targets['y']).square().mean()
            return loss, {'regression': loss}
    def adapt(rows):
        return TrainingBatch({'x': torch.tensor(rows, dtype=torch.float32)}, {'y': torch.zeros(len(rows))})
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
    engine = TrainingEngine(model, optimizer, tasks=Tasks(), scheduler=scheduler)
    completed = list(training_iterations(engine, [1.] * 5, adapt, stop_step=4, batch_size=2))
    assert [iteration.result.total for iteration in completed] == pytest.approx([1., .64])
    assert model.weight.item() == pytest.approx(.72)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(.025)
    assert scheduler.last_epoch == 2


def test_completed_iteration_releases_the_adapted_batch_before_runner_callbacks():
    import weakref
    references = []
    def adapt(rows):
        batch = TrainingBatch({'x': torch.tensor(rows)}, {})
        references.append(weakref.ref(batch))
        return batch
    engine = SimpleNamespace(step=lambda batch, step: StepResult(0., {}, 0.))
    iterations = training_iterations(engine, [1, 2], adapt, stop_step=2, batch_size=1)
    first = next(iterations)
    assert first.records == (1,)
    assert references[0]() is None
    iterations.close()
