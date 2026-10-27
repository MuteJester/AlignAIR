"""Shared learning-rate schedules validate before allocating optimizer state."""
import pytest
import torch

from alignair.errors import ConfigurationError
from alignair.train import CosineSchedule

pytestmark = pytest.mark.component


@pytest.mark.parametrize('values', [{'total_steps': 0}, {'total_steps': True}, {'total_steps': 1.2},
    {'total_steps': 5, 'min_lr': -1}, {'total_steps': 5, 'min_lr': True},
    {'total_steps': 5, 'min_lr': float('nan')}, {'total_steps': 5, 'min_lr': 'small'}])
def test_invalid_cosine_policy_is_rejected(values):
    with pytest.raises(ConfigurationError):
        CosineSchedule(**values)


@pytest.mark.parametrize('lr', [True, float('inf'), 'large', .000001])
def test_initial_lr_cannot_be_below_floor_or_invalid(lr):
    with pytest.raises(ConfigurationError):
        CosineSchedule(10).validate(lr)


def test_cosine_schedule_matches_declared_initial_and_terminal_rates():
    policy = CosineSchedule(4, .01)
    policy.validate(.1)
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=.1)
    scheduler = policy.build(optimizer)
    values = [optimizer.param_groups[0]['lr']]
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        values.append(optimizer.param_groups[0]['lr'])
    assert values[0] == .1 and values[-1] == .01
    assert all(first > second for first, second in zip(values, values[1:]))
    assert policy.to_dict() == {'policy': 'cosine-v1', 'total_steps': 4, 'min_lr': .01}
