"""Sigmoid scaling, top-1 binary grid fitting and right-closed ECE bins."""
import numpy as np
import pytest

from alignair.calibration import apply_temperature, ece, fit_temperature
from alignair.errors import ConfigurationError, InputError, NumericalError

pytestmark = pytest.mark.component


def test_scaling_matches_hand_calculated_odds_and_retains_identity_paths():
    probabilities = np.array([.2, .5, .8])
    np.testing.assert_allclose(apply_temperature(probabilities, .5),
                               [1 / 17, .5, 16 / 17], rtol=1e-14)
    assert apply_temperature(probabilities, None) is probabilities
    assert apply_temperature(probabilities, 1.) is probabilities


def test_boundary_clipping_and_extreme_positive_temperatures_remain_finite():
    with np.errstate(all='raise'):
        scaled = apply_temperature(np.array([0., 1.]), .5)
        saturated = apply_temperature(np.array([.1, .9]), 1e-300)
    assert 0 < scaled[0] < 1e-10 and .999999 < scaled[1] < 1
    np.testing.assert_array_equal(saturated, [0., 1.])


def test_integer_inputs_do_not_truncate_transformed_probabilities():
    scaled = apply_temperature(np.array([0, 1]), .5)
    assert scaled.dtype == np.float64 and scaled[1] > .99


def test_clipping_can_merge_ranks_and_fixed_threshold_membership_can_change():
    probabilities = np.array([1e-9, 2e-9])
    assert probabilities.argmax() == 1
    assert apply_temperature(probabilities, .5).argmax() == 0
    row = np.array([.9, .3])
    assert (row >= .2).tolist() == [True, True]
    assert (apply_temperature(row, .5) >= .2).tolist() == [True, False]


def test_grid_fit_selects_known_boundary_optima_and_keeps_first_equal_minimum():
    assert fit_temperature([.8, .8], [1, 0], lo=.5, hi=2., n=3) == 2.
    assert fit_temperature([.8, .8], [1, 1], lo=.5, hi=2., n=3) == .5
    assert fit_temperature([.5, .5], [1, 0]) == .2


def test_ece_uses_right_closed_bins_with_zero_and_one_included():
    assert ece([0., .5, .75, 1.], [1, 0, 1, 1], n_bins=2) == .1875
    assert ece([], []) == 0.


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_floating_dtype_and_original_values_are_preserved(dtype):
    row = np.array([.2, .3], dtype=dtype)
    before = row.copy()
    assert apply_temperature(row, .7).dtype == dtype
    np.testing.assert_array_equal(row, before)


@pytest.mark.parametrize('temperature', [0, -1, True, float('nan'), float('inf'), '.5', 2 ** 10000])
def test_invalid_temperatures_raise_contextual_configuration_errors(temperature):
    with pytest.raises(ConfigurationError) as caught:
        apply_temperature([.5], temperature)
    assert caught.value.context == {'stage': 'temperature_calibration', 'field': 'temperature'}


@pytest.mark.parametrize('probabilities', [[-.1], [1.1], [True, False], ['.5'], [[.5], [.1, .9]]])
def test_invalid_probabilities_are_rejected(probabilities):
    with pytest.raises(InputError):
        apply_temperature(probabilities, .5)


@pytest.mark.parametrize('probabilities', [[float('nan')], [float('inf')]])
def test_nonfinite_probabilities_raise_numerical_errors(probabilities):
    with pytest.raises(NumericalError):
        apply_temperature(probabilities, 1.)


@pytest.mark.parametrize('fields', [{'lo': 0}, {'hi': -1}, {'lo': 2., 'hi': 1.}, {'n': 0}, {'n': True}])
def test_invalid_grid_configuration_is_rejected(fields):
    with pytest.raises(ConfigurationError):
        fit_temperature([.8], [1], **fields)


@pytest.mark.parametrize('probabilities,correct', [([], []), ([[.5]], [1]), ([.5], []),
    ([.5], [2]), ([.5], [float('nan')]), ([.5], [[1], [0, 1]])])
def test_grid_fit_requires_aligned_observed_binary_correctness(probabilities, correct):
    with pytest.raises(InputError):
        fit_temperature(probabilities, correct)


def test_ece_requires_positive_integer_bin_count():
    with pytest.raises(ConfigurationError):
        ece([.5], [1], n_bins=0)
