"""Training-state construction: placement, identity and ownership; the optimizer is AdamW and the scheduler the
run's ``CosineSchedule``."""
from types import SimpleNamespace

import pytest
import torch

from alignair.errors import ConfigurationError, current_error_context
from alignair.train import CosineSchedule
from alignair.train.setup import build_training_engine

pytestmark = pytest.mark.component


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.))
        self.other = torch.nn.Parameter(torch.tensor(0.))
        self.register_buffer('offset', torch.tensor(1.))
    def forward(self, inputs):
        return self.value


def plain_tasks():
    return SimpleNamespace(loss=lambda output, targets: (output, {'linear': output}))


def test_default_adamw_binds_the_models_parameters_and_preserves_objects():
    model = ScalarModel().eval()
    tasks = plain_tasks()
    engine = build_training_engine(model, tasks=tasks, lr=.01)
    assert engine.model is model and engine.tasks is tasks
    assert not model.training
    assert type(engine.optimizer) is torch.optim.AdamW
    assert [id(p) for p in engine.optimizer.param_groups[0]['params']] == [id(p) for p in model.parameters()]
    assert engine.optimizer.defaults['weight_decay'] == .01  # preserve the established AdamW defaults
    assert engine.optimizer.param_groups[0]['lr'] == .01
    assert engine.scheduler is None
    engine.step(({}, {}))
    assert model.value.item() < 0


def test_a_shared_parameter_is_optimized_once():
    model = ScalarModel()
    model.sub = torch.nn.Module()
    model.sub.register_parameter('shared', model.value)
    engine = build_training_engine(model, tasks=plain_tasks())
    assert [id(p) for p in engine.optimizer.param_groups[0]['params']] == [id(model.value), id(model.other)]


def test_a_cosine_schedule_advances_once_after_the_update():
    model = ScalarModel()
    engine = build_training_engine(model, tasks=plain_tasks(), lr=1., schedule=CosineSchedule(total_steps=2, min_lr=0.))
    engine.step(({},{}))
    assert engine.scheduler.last_epoch == 1 and engine.optimizer.param_groups[0]['lr'] == pytest.approx(.5)


def test_placement_occurs_before_optimizer_creation_and_preserves_current_device_by_default():
    model = ScalarModel()
    old_parameter = model.value
    engine = build_training_engine(model, tasks=plain_tasks(), device='meta')
    assert model.value is not old_parameter
    assert model.offset.device.type == 'meta'
    assert engine.optimizer.param_groups[0]['params'][0] is model.value
    second = build_training_engine(model, tasks=plain_tasks())
    assert second.optimizer.param_groups[0]['params'][0].device.type == 'meta'


def test_a_model_without_parameters_is_rejected():
    with pytest.raises(ConfigurationError, match='at least one trainable'):
        build_training_engine(torch.nn.Identity(), tasks=plain_tasks())


def test_frozen_parameters_retain_their_order_in_the_default_group():
    model = ScalarModel()
    model.register_parameter('frozen',torch.nn.Parameter(torch.tensor(4.),requires_grad=False))
    tasks = SimpleNamespace(loss=lambda output,*_: (output+1,{}))
    default = build_training_engine(model,tasks=tasks)
    assert default.optimizer.param_groups[0]['params'][2] is model.frozen


@pytest.mark.parametrize('lr',[0,-.1,float('nan'),float('inf'),True,'0.1'])
def test_bad_learning_rate_fails_before_module_placement(lr,monkeypatch):
    model=ScalarModel()
    def forbidden(*args):raise AssertionError('placement preceded preflight')
    monkeypatch.setattr(model,'to',forbidden)
    with pytest.raises(ConfigurationError) as caught:
        build_training_engine(model,tasks=plain_tasks(),lr=lr)
    assert caught.value.context['field']=='lr' and caught.value.context['stage']=='training_setup'
    assert current_error_context()=={}


@pytest.mark.parametrize('options', [{'grad_clip':False},{'schedule':'cosine'},{'device':'unknown-device'}])
def test_bad_initialization_options_are_rejected(options):
    with pytest.raises(ConfigurationError):
        build_training_engine(ScalarModel(),tasks=plain_tasks(),**options)


def test_models_with_mixed_devices_require_an_explicit_placement_policy():
    model=ScalarModel()
    model.register_buffer('elsewhere',torch.zeros(1,device='meta'))
    with pytest.raises(ConfigurationError,match='multiple devices'):
        build_training_engine(model,tasks=plain_tasks())


@pytest.mark.parametrize('model,tasks',[(object(),None),(ScalarModel(),None),(ScalarModel(),object())])
def test_invalid_model_and_task_contracts_are_domain_errors(model,tasks):
    with pytest.raises(ConfigurationError):
        build_training_engine(model,tasks=tasks)


def test_completely_frozen_state_is_rejected_before_optimizer_construction():
    model=ScalarModel().requires_grad_(False)
    with pytest.raises(ConfigurationError,match='at least one trainable'):
        build_training_engine(model,tasks=plain_tasks())


def test_device_transfer_cannot_silently_split_shared_parameter_aliases():
    model = ScalarModel()
    model.sub = torch.nn.Module()
    model.sub.register_parameter('tied', model.value)
    with pytest.raises(ConfigurationError, match='shared parameter aliases'):
        build_training_engine(model, tasks=plain_tasks(), device='meta')


def test_reused_submodules_keep_their_identity_during_placement():
    model = torch.nn.Module()
    model.layer = torch.nn.Linear(2, 2)
    model.alias = model.layer
    engine = build_training_engine(model, tasks=SimpleNamespace(loss=lambda *_: None), device='meta')
    assert model.layer is model.alias
    assert len(engine.optimizer.param_groups[0]['params']) == 2
    assert all(parameter.device.type=='meta' for parameter in engine.optimizer.param_groups[0]['params'])
