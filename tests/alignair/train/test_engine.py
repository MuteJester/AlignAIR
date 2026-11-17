"""Shared optimizer contracts: clipping, scheduling, rejected steps and validation."""
from types import SimpleNamespace

import pytest
import torch

from alignair.data import TrainingBatch
from alignair.errors import ConfigurationError
from alignair.tasks import MetricReport
from alignair.core.runtime import evaluation_mode
from alignair.train.engine import TrainingEngine
from alignair.train.guards import NonFiniteTrainingError

pytestmark = pytest.mark.component


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.))
        self.other = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, inputs):
        return self.value


def test_step_clips_all_optimized_parameters_and_advances_the_scheduler():
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
    tasks = SimpleNamespace(loss=lambda out, _: (3*out + 4*model.other, {'linear': out}))
    engine = TrainingEngine(model, optimizer, tasks=tasks, grad_clip=1., scheduler=scheduler)
    result = engine.step(TrainingBatch({}, {}), step=1)
    assert result.gradient_norm == pytest.approx(5.)
    assert model.value.item() == pytest.approx(-.6) and model.other.item() == pytest.approx(-.8)
    assert optimizer.param_groups[0]['lr'] == .5
    assert scheduler.last_epoch == 1
    assert result.total == 0. and result.parts == {'linear': 0.}


@pytest.mark.parametrize('failure', ['parameter_gradient', 'loss', 'norm_overflow'])
def test_rejected_step_does_not_update_parameters_optimizer_or_scheduler(failure):
    model = ScalarModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    def objective(out, _):
        if failure == 'loss':
            return out + float('nan'), {}
        if failure == 'norm_overflow':
            return (out + model.other) * 1e30, {}
        return out.square() + model.other.abs().sqrt(), {}
    engine = TrainingEngine(model, optimizer, tasks=SimpleNamespace(loss=objective), grad_clip=1., scheduler=scheduler)
    with pytest.raises(NonFiniteTrainingError) as caught:
        engine.step(({}, {}), step=17)
    assert caught.value.context['step'] == 17
    if failure == 'parameter_gradient':
        assert caught.value.context['parameters'] == ['model.other']
        assert caught.value.context['reason'] == 'nonfinite_gradient'
    elif failure == 'norm_overflow':
        assert caught.value.context['parameters'] == []
        assert caught.value.context['reason'] == 'gradient_norm_overflow'
    assert model.value.item() == model.other.item() == 0
    assert not optimizer.state
    assert scheduler.last_epoch == 0
    assert all(p.grad is None for group in optimizer.param_groups for p in group['params'])


@pytest.mark.parametrize('clip', [0, -1., float('inf'), float('nan'), True, '1'])
def test_invalid_clipping_is_rejected_at_construction(clip):
    model = ScalarModel()
    with pytest.raises(ConfigurationError, match='grad_clip'):
        TrainingEngine(model, torch.optim.SGD(model.parameters(), lr=.1), tasks=object(), grad_clip=clip)


def test_a_scheduler_of_another_optimizer_is_a_configuration_error():
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    other = torch.optim.SGD(model.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.StepLR(other, 1)
    with pytest.raises(ConfigurationError, match='scheduler'):
        TrainingEngine(model, optimizer, tasks=object(), scheduler=scheduler)


@pytest.mark.parametrize('fail', [False, True])
def test_evaluation_restores_mixed_module_modes_and_gradient_state(fail):
    model = torch.nn.Sequential(torch.nn.Dropout(), torch.nn.Sequential(torch.nn.Dropout()))
    model.train()
    model[1].eval()
    modes = [module.training for module in model.modules()]
    def evaluate():
        with evaluation_mode(model):
            assert not any(module.training for module in model.modules())
            assert not torch.is_grad_enabled()
            if fail:
                raise RuntimeError('evaluation failed')
    if fail:
        with pytest.raises(RuntimeError, match='evaluation failed'):
            evaluate()
    else:
        evaluate()
    assert [module.training for module in model.modules()] == modes
    assert torch.is_grad_enabled()


def test_engine_validation_uses_task_support_and_accepts_custom_models_without_config():
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    def report(outputs, targets):
        assert not model.training and not torch.is_grad_enabled()
        return MetricReport(totals={'accuracy': targets['hits']}, counts={'accuracy': targets['count']})
    tasks = SimpleNamespace(metric_report=report, selection_score=lambda values: values['accuracy'])
    engine = TrainingEngine(model, optimizer, tasks=tasks)
    result = engine.validate([({}, {'hits': 1., 'count': 2.}), ({}, {'hits': 1., 'count': 1.})])
    assert result.metrics == {'accuracy': pytest.approx(2/3)}
    assert result.score == pytest.approx(2/3)
    assert model.training


@pytest.mark.parametrize('constant', [False, True])
def test_an_optimizer_disconnected_from_the_loss_fails_before_a_false_successful_step(constant):
    from alignair.errors import TrainingError
    model = ScalarModel()
    unrelated = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.SGD([unrelated], lr=.1)
    tasks = SimpleNamespace(loss=lambda out, *_: ((out.detach() if constant else out)+1, {}))
    engine = TrainingEngine(model, optimizer, tasks=tasks)
    with pytest.raises(TrainingError, match='no gradients|no gradient graph'):
        engine.step(({}, {}))
    assert unrelated.item() == 1.
