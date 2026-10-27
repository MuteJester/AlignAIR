"""Training-state construction: placement, identity, ownership, and independent factories."""
from types import SimpleNamespace

import pytest
import torch

from alignair.errors import ConfigurationError, current_error_context
from alignair.neural.weighting import UncertaintyWeight
from alignair.train import build_training_engine

pytestmark = pytest.mark.component


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.))
        self.register_buffer('offset', torch.tensor(1.))
    def forward(self, inputs):
        return self.value


def tasks_with_weight():
    def loss(output, targets, weights):
        return output + 2*weights['extra'].log_var, {'linear': output}
    return SimpleNamespace(make_weights=lambda: torch.nn.ModuleDict({'extra': UncertaintyWeight()}), loss=loss)


def test_default_adamw_binds_model_then_supplied_weights_and_preserves_objects():
    model = ScalarModel().eval()
    tasks = tasks_with_weight()
    weights = tasks.make_weights()
    engine = build_training_engine(model, tasks=tasks, weights=weights, lr=.01)
    assert engine.model is model and engine.tasks is tasks and engine.weights is weights
    assert not model.training
    assert type(engine.optimizer) is torch.optim.AdamW
    expected = [*model.parameters(), *weights.parameters()]
    assert [id(p) for p in engine.optimizer.param_groups[0]['params']] == [id(p) for p in expected]
    assert engine.optimizer.defaults['weight_decay'] == .01  # preserve the established AdamW defaults
    assert engine.optimizer.param_groups[0]['lr'] == .01
    assert engine.scheduler is None
    engine.step(({}, {}))
    assert model.value.item() < 0 and weights['extra'].log_var.item() < 0


def test_task_weights_are_created_once_when_not_supplied():
    tasks = tasks_with_weight()
    original = tasks.make_weights
    calls = []
    def make():
        calls.append(1)
        return original()
    tasks.make_weights = make
    engine = build_training_engine(ScalarModel(), tasks=tasks)
    assert calls == [1]
    engine.step(({}, {}))
    assert calls == [1] and 'extra' in engine.weights


def test_explicit_empty_weight_mapping_needs_no_weight_factory():
    weights = {}
    tasks = SimpleNamespace(loss=lambda output,*_: (output+1, {}))
    engine = build_training_engine(ScalarModel(), tasks=tasks, weights=weights)
    assert engine.weights is weights
    assert engine.step(({},{})).gradient_norm == 1.


def test_plain_weight_mappings_and_shared_parameters_are_not_copied_or_optimized_twice():
    model = ScalarModel()
    weight = torch.nn.Module()
    weight.register_parameter('shared', model.value)
    weights = {'shared': weight, 'alias': weight}
    tasks = SimpleNamespace(loss=lambda output,*_: (output+1,{}))
    engine = build_training_engine(model, tasks=tasks, weights=weights)
    assert engine.weights is weights
    assert len(engine.optimizer.param_groups[0]['params']) == 1
    assert engine.optimizer.param_groups[0]['params'][0] is model.value


def test_module_dict_root_parameters_are_included_in_the_optimizer():
    weights = torch.nn.ModuleDict()
    weights.register_parameter('global_weight', torch.nn.Parameter(torch.tensor(0.)))
    engine = build_training_engine(ScalarModel(), weights=weights,
        tasks=SimpleNamespace(loss=lambda output,_,weights: (output+weights.global_weight,{})))
    assert any(p is weights.global_weight for p in engine.optimizer.param_groups[0]['params'])
    engine.step(({},{}))
    assert weights.global_weight.item() < 0


def test_custom_optimizer_and_scheduler_keep_the_hand_calculated_update_order():
    model = ScalarModel()
    weights = tasks_with_weight().make_weights()
    def optimizer_factory(parameters, *, lr):
        return torch.optim.SGD(parameters, lr=lr)
    engine = build_training_engine(model, tasks=tasks_with_weight(), weights=weights, lr=1.,
        optimizer_factory=optimizer_factory,
        scheduler_factory=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=.5))
    engine.step(({},{}))
    assert model.value.item() == -1 and weights['extra'].log_var.item() == -2
    assert engine.optimizer.param_groups[0]['lr'] == .5 and engine.scheduler.last_epoch == 1


def test_placement_occurs_before_optimizer_creation_and_preserves_current_device_by_default():
    model = ScalarModel()
    old_parameter = model.value
    weights = tasks_with_weight().make_weights()
    observations = []
    def optimizer(parameters, *, lr):
        observations.append(tuple(p.device.type for p in parameters))
        return torch.optim.SGD(parameters, lr=lr)
    engine = build_training_engine(model, tasks=tasks_with_weight(), weights=weights, device='meta',
                                   optimizer_factory=optimizer)
    assert model.value is not old_parameter
    assert observations == [('meta','meta')]
    assert model.offset.device.type == 'meta' and weights['extra'].log_var.device.type == 'meta'
    assert engine.optimizer.param_groups[0]['params'][0] is model.value
    second = build_training_engine(model, tasks=tasks_with_weight())
    assert second.weights['extra'].log_var.device.type == 'meta'


def test_tensor_free_model_uses_cpu_and_can_optimize_task_weights_only():
    model = torch.nn.Identity()
    tasks = SimpleNamespace(make_weights=tasks_with_weight().make_weights,
                            loss=lambda output,_,weights: (weights['extra'].log_var,{}))
    engine = build_training_engine(model,tasks=tasks)
    assert engine.weights['extra'].log_var.device.type == 'cpu'
    engine.step((torch.tensor(0.),{}))
    assert engine.weights['extra'].log_var.item() < 0


def test_frozen_parameters_retain_default_order_but_custom_groups_may_omit_them():
    model = ScalarModel()
    model.register_parameter('frozen',torch.nn.Parameter(torch.tensor(4.),requires_grad=False))
    tasks = SimpleNamespace(loss=lambda output,*_: (output+1,{}))
    default = build_training_engine(model,tasks=tasks,weights={})
    assert default.optimizer.param_groups[0]['params'][1] is model.frozen
    filtered = build_training_engine(model,tasks=tasks,weights={},optimizer_factory=
        lambda parameters,lr: torch.optim.SGD([p for p in parameters if p.requires_grad],lr=lr))
    assert len(filtered.optimizer.param_groups[0]['params']) == 1


@pytest.mark.parametrize('lr',[0,-.1,float('nan'),float('inf'),True,'0.1'])
def test_bad_learning_rate_fails_before_module_placement(lr,monkeypatch):
    model=ScalarModel()
    def forbidden(*args):raise AssertionError('placement preceded preflight')
    monkeypatch.setattr(model,'to',forbidden)
    with pytest.raises(ConfigurationError) as caught:
        build_training_engine(model,tasks=tasks_with_weight(),lr=lr)
    assert caught.value.context['field']=='lr' and caught.value.context['stage']=='training_setup'
    assert current_error_context()=={}


@pytest.mark.parametrize('options', [{'grad_clip':False},{'optimizer_factory':None},{'scheduler_factory':False},
                                     {'device':'unknown-device'}])
def test_bad_initialization_options_are_rejected(options):
    with pytest.raises(ConfigurationError):
        build_training_engine(ScalarModel(),tasks=tasks_with_weight(),**options)


def test_models_with_mixed_devices_require_an_explicit_placement_policy():
    model=ScalarModel()
    model.register_buffer('other',torch.zeros(1,device='meta'))
    with pytest.raises(ConfigurationError,match='multiple devices'):
        build_training_engine(model,tasks=tasks_with_weight())


@pytest.mark.parametrize('model,tasks,weights',[(object(),None,None),(ScalarModel(),None,None),
    (ScalarModel(),object(),{}),(ScalarModel(),SimpleNamespace(loss=lambda *_:None),None),
    (ScalarModel(),tasks_with_weight(),[]),(ScalarModel(),tasks_with_weight(),{'bad':None})])
def test_invalid_model_task_and_weight_contracts_are_domain_errors(model,tasks,weights):
    with pytest.raises(ConfigurationError):
        build_training_engine(model,tasks=tasks,weights=weights)


def test_completely_frozen_state_is_rejected_before_optimizer_construction():
    model=ScalarModel().requires_grad_(False)
    with pytest.raises(ConfigurationError,match='at least one trainable'):
        build_training_engine(model,tasks=tasks_with_weight(),weights={})


@pytest.mark.parametrize('kind',['not_optimizer','missing_model','missing_weight','foreign','duplicate'])
def test_custom_optimizer_cannot_silently_change_parameter_ownership(kind):
    model=ScalarModel()
    weights=tasks_with_weight().make_weights()
    def make(parameters,*,lr):
        if kind=='not_optimizer':return object()
        if kind=='missing_model':parameters=parameters[1:]
        elif kind=='missing_weight':parameters=parameters[:1]
        elif kind=='foreign':parameters=(*parameters,torch.nn.Parameter(torch.tensor(1.)))
        optimizer=torch.optim.SGD(parameters,lr=lr)
        if kind=='duplicate':optimizer.param_groups[0]['params'].append(parameters[0])
        return optimizer
    with pytest.raises(ConfigurationError,match='optimizer'):
        build_training_engine(model,tasks=tasks_with_weight(),weights=weights,optimizer_factory=make)


@pytest.mark.parametrize('kind',['none','missing_step','wrong_optimizer','metric'])
def test_custom_scheduler_must_bind_the_same_optimizer(kind):
    def scheduler(optimizer):
        if kind=='none':return None
        if kind=='metric':return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        if kind=='missing_step':return SimpleNamespace(optimizer=optimizer)
        other=torch.optim.SGD(optimizer.param_groups[0]['params'],lr=.1)
        return torch.optim.lr_scheduler.StepLR(other,1)
    with pytest.raises(ConfigurationError,match='scheduler_factory'):
        build_training_engine(ScalarModel(),tasks=tasks_with_weight(),scheduler_factory=scheduler)


def test_default_tasks_are_selected_from_the_shared_model_configuration():
    from alignair.core import AlignAIR,AlignAIRConfig
    cfg=AlignAIRConfig(2,2,has_d=False,max_seq_length=128,embed_dim=4,filters=4,block_out=8).with_amino_acid_heads()
    model=AlignAIR(cfg)
    engine=build_training_engine(model)
    assert engine.tasks.config is cfg and len(engine.weights)==0


def test_device_transfer_cannot_silently_split_shared_parameter_aliases():
    model = ScalarModel()
    weight = torch.nn.Module()
    weight.register_parameter('tied', model.value)
    def forbidden(*args, **kwargs):
        raise AssertionError('optimizer was created for split aliases')
    with pytest.raises(ConfigurationError, match='shared parameter aliases'):
        build_training_engine(model, tasks=tasks_with_weight(), weights={'shared':weight},
                              device='meta', optimizer_factory=forbidden)


def test_reused_submodules_keep_their_identity_during_placement():
    model = torch.nn.Module()
    model.layer = torch.nn.Linear(2, 2)
    model.alias = model.layer
    weights = {'same_layer':model.layer}
    engine = build_training_engine(model, tasks=SimpleNamespace(loss=lambda *_: None),
                                   weights=weights, device='meta')
    assert model.layer is model.alias is weights['same_layer']
    assert len(engine.optimizer.param_groups[0]['params']) == 2
    assert all(parameter.device.type=='meta' for parameter in engine.optimizer.param_groups[0]['params'])


@pytest.mark.parametrize('config',[None,{}])
def test_custom_configurations_require_explicit_task_selection(config):
    model=ScalarModel()
    model.cfg=config
    with pytest.raises(ConfigurationError,match='AlignAIRConfig'):
        build_training_engine(model)
    assert build_training_engine(model,tasks=tasks_with_weight()).model is model
