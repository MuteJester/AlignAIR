"""Hand-computed categorical scores, aligned observation contracts and bin boundaries."""
import math

import numpy as np
import pytest

from alignair.core.probabilities import validate_categorical_probabilities, validate_categorical_targets
from alignair.errors import ConfigurationError, InputError, NumericalError
from alignair.evaluate import categorical_nll, frequency_accuracy_report, multiclass_brier_score

pytestmark = pytest.mark.component


def test_probability_scores_match_explicit_two_class_residuals_and_log_probabilities():
    probabilities = np.array([[.8, .2], [.1, .9]])
    assert categorical_nll(probabilities, [0, 0]) == pytest.approx(-math.log(.08) / 2)
    assert multiclass_brier_score(probabilities, [0, 0]) == pytest.approx(.85)
    assert multiclass_brier_score([[.5, .5], [.5, .5]], [0, 1]) == .5
    assert categorical_nll([[1, 0], [0, 1]], [0, 1]) == 0.
    assert multiclass_brier_score([[1, 0], [0, 1]], [0, 1]) == 0.


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_zero_true_probability_has_a_finite_clipped_nll_at_every_supported_precision(dtype):
    with np.errstate(all='raise'):
        value = categorical_nll(np.array([[0., 1.]], dtype=dtype), [0])
    assert math.isfinite(value) and value == pytest.approx(-math.log(1e-12), rel=1e-6)


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.uint8])
@pytest.mark.parametrize('layout', ['c', 'f', 'view'])
def test_scores_preserve_readonly_input_values_dtype_and_layout(dtype, layout):
    probabilities = np.array([[1, 0], [0, 1]], dtype=dtype)
    if layout == 'f': probabilities = np.asfortranarray(probabilities)
    elif layout == 'view': probabilities = np.repeat(probabilities, 2, axis=1)[:, ::2]
    probabilities.flags.writeable = False
    before = probabilities.copy()
    assert multiclass_brier_score(probabilities, [1, 0]) == 2.
    assert math.isfinite(categorical_nll(probabilities, [1, 0]))
    np.testing.assert_array_equal(probabilities, before)
    assert probabilities.dtype == dtype and not probabilities.flags.writeable


def test_shared_probability_validation_keeps_arrays_and_supports_prediction_vectors():
    values = np.array([[.5, .3, .2]], dtype=np.float32)
    assert validate_categorical_probabilities(values) is values
    np.testing.assert_array_equal(validate_categorical_probabilities(values[0], 1), values[0])
    np.testing.assert_array_equal(validate_categorical_targets([0], values.shape), [0])


@pytest.mark.parametrize('dimensions', [0, 3, True])
def test_invalid_probability_dimensions_are_configuration_errors(dimensions):
    with pytest.raises(ConfigurationError):
        validate_categorical_probabilities([1.], dimensions)


@pytest.mark.parametrize('values', [[], np.empty((0, 2)), [.5, .5], [[.1, .2]], [[-.1, 1.1]],
    [[True, False]], [['.5', '.5']], [[.5], [.2, .8]], [[.5+0j, .5+0j]]])
@pytest.mark.parametrize('score', [categorical_nll, multiclass_brier_score])
def test_malformed_probability_matrices_are_rejected(values, score):
    with pytest.raises(InputError) as caught:
        score(values, [0])
    assert caught.value.context['stage'] == 'probability_evaluation'


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_probabilities_have_numerical_errors(value):
    with pytest.raises(NumericalError):
        categorical_nll([[value, .5]], [0])


@pytest.mark.parametrize('truth', [[], [-1], [2], [True], [0.], [[0]], [[0], [0, 1]]])
def test_truth_must_be_an_aligned_observed_integer_id_vector(truth):
    with pytest.raises(InputError):
        multiclass_brier_score([[.5, .5]], truth)


def test_frequency_report_preserves_left_closed_bins_last_boundary_and_empty_bins():
    report = frequency_accuracy_report([0, 0, 1, 1], [0, 1, 1, 0], [0., .25, .5, 1.], (0., .25, .5, .75, 1.))
    assert report == {'[0.0,0.25)': {'n': 1, 'acc': 1.}, '[0.25,0.5)': {'n': 1, 'acc': 0.},
                      '[0.5,0.75)': {'n': 1, 'acc': 1.}, '[0.75,1.0]': {'n': 1, 'acc': 0.}}
    assert frequency_accuracy_report([], [], [], (0, 1)) == {'[0,1]': {'n': 0, 'acc': None}}
    assert frequency_accuracy_report([0], [0], [.9], (0., .5)) == {'[0.0,0.5]': {'n': 0, 'acc': None}}


def test_frequency_report_retains_dtype_sensitive_edge_comparisons():
    frequency = np.array([.02], dtype=np.float16)
    assert frequency_accuracy_report([0], [0], frequency, (0., .02)) == {'[0.0,0.02]': {'n': 1, 'acc': 1.}}


@pytest.mark.parametrize('frequencies', [[], [-.1], [1.1], [float('nan')], [[.5]], ['bad'], ['.5']])
def test_invalid_or_unaligned_frequencies_are_rejected(frequencies):
    with pytest.raises(InputError):
        frequency_accuracy_report([0], [0], frequencies, (0, 1))


@pytest.mark.parametrize('truth,predicted', [([-1], [0]), ([0], [True]), ([0], []), ([[0]], [0])])
def test_frequency_report_uses_shared_classification_id_contract(truth, predicted):
    with pytest.raises(InputError):
        frequency_accuracy_report(truth, predicted, [.5], (0, 1))


@pytest.mark.parametrize('edges', [[], [0], [0, 0], [-1, 1], [0, 2], [0, float('nan')],
                                  [True, False], [[0, 1]], [[0], [1, 2]]])
def test_invalid_frequency_bin_configuration_is_rejected(edges):
    with pytest.raises(ConfigurationError):
        frequency_accuracy_report([0], [0], [.5], edges)
