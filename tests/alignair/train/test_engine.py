"""Shared optimizer contracts, including parameters that live outside the model."""
from types import SimpleNamespace

import pytest
import torch

from alignair.data import TrainingBatch
from alignair.errors import ConfigurationError
from alignair.neural.weighting import UncertaintyWeight
from alignair.tasks import MetricReport
from alignair.train import TrainingEngine, evaluation_mode
from alignair.train.guards import NonFiniteLossError

pytestmark = pytest.mark.component


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, inputs):
        return self.value


def test_step_clips_all_optimized_parameters_then_constrains_and_advances_scheduler():
    model = ScalarModel()
    weights = torch.nn.ModuleDict({'extra': UncertaintyWeight(min_log_var=-.5)})
    optimizer = torch.optim.SGD([*model.parameters(), *weights.parameters()], lr=1.)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
    tasks = SimpleNamespace(loss=lambda out, _, weights: (3*out + 4*weights['extra'].log_var, {'linear': out}))
    engine = TrainingEngine(model, optimizer, tasks=tasks, weights=weights, grad_clip=1., scheduler=scheduler)
    result = engine.step(TrainingBatch({}, {}), step=1)
    assert result.gradient_norm == pytest.approx(5.)
    assert model.value.item() == pytest.approx(-.6)
    assert weights['extra'].log_var.item() == pytest.approx(-.5)  # clamp follows the clipped update
    assert optimizer.param_groups[0]['lr'] == .5
    assert scheduler.last_epoch == 1
    assert result.total == 0. and result.parts == {'linear': 0.}


@pytest.mark.parametrize('failure', ['task_gradient', 'loss', 'norm_overflow'])
def test_rejected_step_does_not_update_weights_optimizer_or_scheduler(failure):
    model = ScalarModel()
    weights = torch.nn.ModuleDict({'extra': UncertaintyWeight()})
    optimizer = torch.optim.AdamW([*model.parameters(), *weights.parameters()], lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    def objective(out, _, weights):
        if failure == 'loss':
            return out + float('nan'), {}
        if failure == 'norm_overflow':
            return (out + weights['extra'].log_var) * 1e30, {}
        return out.square() + weights['extra'].log_var.abs().sqrt(), {}
    engine = TrainingEngine(model, optimizer, tasks=SimpleNamespace(loss=objective), weights=weights,
                            grad_clip=1., scheduler=scheduler)
    with pytest.raises(NonFiniteLossError) as caught:
        engine.step(({}, {}), step=17)
    assert caught.value.context['step'] == 17
    if failure == 'task_gradient':
        assert caught.value.context['parameters'] == ['tasks.extra.log_var']
        assert caught.value.context['reason'] == 'nonfinite_gradient'
    elif failure == 'norm_overflow':
        assert caught.value.context['parameters'] == []
        assert caught.value.context['reason'] == 'gradient_norm_overflow'
    assert model.value.item() == weights['extra'].log_var.item() == 0
    assert not optimizer.state
    assert scheduler.last_epoch == 0
    assert all(p.grad is None for group in optimizer.param_groups for p in group['params'])


@pytest.mark.parametrize('clip', [0, -1., float('inf'), float('nan'), True, '1'])
def test_invalid_clipping_is_rejected_at_construction(clip):
    model = ScalarModel()
    with pytest.raises(ConfigurationError, match='grad_clip'):
        TrainingEngine(model, torch.optim.SGD(model.parameters(), lr=.1), tasks=object(), weights={}, grad_clip=clip)


def test_missing_task_parameters_and_wrong_scheduler_are_explicit_configuration_errors():
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    with pytest.raises(ConfigurationError, match='every trainable task weight'):
        TrainingEngine(model, optimizer, tasks=object(), weights={'extra': UncertaintyWeight()})
    other = torch.optim.SGD(model.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.StepLR(other, 1)
    with pytest.raises(ConfigurationError, match='scheduler'):
        TrainingEngine(model, optimizer, tasks=object(), weights={}, scheduler=scheduler)


def test_custom_task_weights_need_no_constraint_hook_and_frozen_weights_need_no_optimizer_slot():
    model = ScalarModel()
    weight = torch.nn.Linear(1, 1, bias=False)
    weight.requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    tasks = SimpleNamespace(loss=lambda out, *_: (out + 1, {}))
    engine = TrainingEngine(model, optimizer, tasks=tasks, weights={'frozen': weight})
    assert engine.step(({}, {})).gradient_norm == 1
    assert model.value.item() == pytest.approx(-.1)


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
    engine = TrainingEngine(model, optimizer, tasks=tasks, weights={})
    result = engine.validate([({}, {'hits': 1., 'count': 2.}), ({}, {'hits': 1., 'count': 1.})])
    assert result.metrics == {'accuracy': pytest.approx(2/3)}
    assert result.score == pytest.approx(2/3)
    assert model.training


def test_historical_trainer_imports_delegate_to_shared_owners():
    from alignair.train import trainer, engine, checkpoints, random_state
    from alignair.aa import resume
    assert trainer.train_step is engine.train_step
    assert trainer.validate is engine.validate
    assert trainer.save_checkpoint is checkpoints.save_checkpoint
    assert trainer._restore_rng is random_state.restore_rng_state
    assert resume.save_checkpoint is checkpoints.save_research_checkpoint


def test_engine_infers_default_tasks_for_a_categorical_model():
    from alignair.core import AlignAIR, AlignAIRConfig
    from alignair.data import CategoricalBatchAdapter
    cfg = AlignAIRConfig(max_seq_length=128, embed_dim=4, filters=4, block_out=8,
                          v_allele_count=2, j_allele_count=2, has_d=False).with_amino_acid_heads()
    model = AlignAIR(cfg)
    batch = CategoricalBatchAdapter.from_config(cfg)([
        {'tokens': [1, 2, 3], 'v_class': 0, 'v_start': 0, 'v_end': 1,
         'j_class': 1, 'j_start': 1, 'j_end': 3}])
    engine = TrainingEngine(model, torch.optim.AdamW(model.parameters(), lr=.001))
    assert engine.step(batch).total > 0
    result = engine.validate([batch])
    assert 0 <= result.score <= 1 and model.training


@pytest.mark.parametrize('constant', [False, True])
def test_an_optimizer_disconnected_from_the_loss_fails_before_a_false_successful_step(constant):
    from alignair.errors import TrainingError
    model = ScalarModel()
    unrelated = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.SGD([unrelated], lr=.1)
    tasks = SimpleNamespace(loss=lambda out, *_: ((out.detach() if constant else out)+1, {}))
    engine = TrainingEngine(model, optimizer, tasks=tasks, weights={})
    with pytest.raises(TrainingError, match='no gradients|no gradient graph'):
        engine.step(({}, {}))
    assert unrelated.item() == 1.


def test_module_dict_root_parameters_are_guarded_and_named_in_gradient_diagnostics():
    model = ScalarModel()
    weights = torch.nn.ModuleDict()
    weights.register_parameter('global_weight',torch.nn.Parameter(torch.tensor(0.)))
    tasks = SimpleNamespace(loss=lambda output,_,weights: (output+weights.global_weight.abs().sqrt(),{}))
    with pytest.raises(ConfigurationError,match='every trainable task weight'):
        TrainingEngine(model,torch.optim.SGD(model.parameters(),lr=.1),tasks=tasks,weights=weights)
    optimizer=torch.optim.SGD([*model.parameters(),*weights.parameters()],lr=.1)
    engine=TrainingEngine(model,optimizer,tasks=tasks,weights=weights)
    with pytest.raises(NonFiniteLossError) as caught:
        engine.step(({},{}))
    assert caught.value.context['parameters']==['tasks.global_weight']
    assert model.value.item()==weights.global_weight.item()==0.


def test_plain_weight_mapping_keeps_named_gradient_diagnostics():
    model=ScalarModel()
    weights={'extra':UncertaintyWeight()}
    optimizer=torch.optim.SGD([*model.parameters(),*weights['extra'].parameters()],lr=.1)
    tasks=SimpleNamespace(loss=lambda output,_,weights: (output+weights['extra'].log_var.abs().sqrt(),{}))
    with pytest.raises(NonFiniteLossError) as caught:
        TrainingEngine(model,optimizer,tasks=tasks,weights=weights).step(({},{}))
    assert caught.value.context['parameters']==['tasks.extra.log_var']
