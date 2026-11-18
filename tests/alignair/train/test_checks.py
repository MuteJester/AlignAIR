"""Training checks its integer, probability and device settings with one rule each."""
import math

import numpy as np
import pytest
import torch

from alignair.errors import ConfigurationError
from alignair.train.checks import device, integer, probability
from alignair.train.random_state import capture_rng_state, same_rng_state

pytestmark = pytest.mark.component


@pytest.mark.parametrize("value", [True, 1.5, "3", None, 0])
def test_an_integer_setting_is_an_int_at_least_its_minimum(value):
    with pytest.raises(ConfigurationError, match="batch_size must be an integer >= 1") as caught:
        integer(value, "batch_size")
    assert caught.value.context["field"] == "batch_size"


def test_an_integer_setting_reads_any_index_and_its_own_minimum():
    assert integer(np.int64(3), "workers") == 3 and type(integer(np.int64(3), "workers")) is int
    assert integer(0, "start_step", 0) == 0
    with pytest.raises(ConfigurationError, match="seed must be an integer >= 0"):
        integer(-1, "seed", 0)


@pytest.mark.parametrize("value", [True, -0.1, 1.01, math.nan, math.inf, "0.5"])
def test_a_probability_setting_is_a_finite_number_in_0_1(value):
    with pytest.raises(ConfigurationError, match="heavy_shm must be a finite number in \\[0, 1\\]"):
        probability(value, "heavy_shm")


def test_a_probability_setting_reads_as_a_float():
    assert probability(1, "heavy_shm") == 1.0 and type(probability(1, "heavy_shm")) is float


def test_a_device_setting_is_a_torch_device():
    assert device("cpu") == torch.device("cpu")
    with pytest.raises(ConfigurationError, match="invalid training device"):
        device("not-a-device")


def test_two_random_states_are_the_same_when_every_generator_is():
    torch.manual_seed(0)
    first = capture_rng_state(include_cuda=False)
    assert same_rng_state(first, capture_rng_state(include_cuda=False))
    torch.rand(1)
    assert not same_rng_state(first, capture_rng_state(include_cuda=False))
