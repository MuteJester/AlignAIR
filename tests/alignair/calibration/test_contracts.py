"""Independent score arithmetic, fitting state transitions and calibration input contracts."""
from copy import deepcopy

import numpy as np
import pytest

from alignair.calibration import Conformal, FrozenConformal, aps_scores, aps_set, conformal_quantile
from alignair.errors import ConfigurationError, InputError, NumericalError

pytestmark = pytest.mark.component


def calibration_data():
    return np.array([[.9, .06, .04], [.9, .06, .04], [.1, .05, .85], [.1, .05, .85]]), np.array([0, 0, 1, 1])


def test_randomized_scores_match_explicit_aps_and_rank_penalty_values():
    probabilities = np.array([[.5, .3, .2], [.1, .7, .2]])
    truth = np.array([1, 0])
    expected = [.7317991932598491, .9683241660290247]
    scores = aps_scores(probabilities, truth, np.random.default_rng(12345))
    np.testing.assert_allclose(scores, expected, rtol=1e-14)
    raps = aps_scores(probabilities, truth, np.random.default_rng(12345), lam=.1, k_reg=1)
    np.testing.assert_allclose(raps, [.8317991932598491, 1.1683241660290247], rtol=1e-14)


def test_quantile_preserves_higher_interpolation_and_small_cohort_clipping():
    assert conformal_quantile(np.array([.1, .2, .3, .4, .5]), .5) == .4
    assert conformal_quantile(np.array([.2]), .01) == .2


def test_prediction_boundaries_and_raps_penalty_keep_the_existing_prefix_rule():
    row = np.array([.5, .3, .2])
    assert aps_set(row, 0.) == aps_set(row, .5) == {0}
    assert aps_set(row, .50001) == {0, 1}
    assert aps_set(row, .85) == {0, 1, 2}
    assert aps_set(row, .85, lam=.1, k_reg=1) == {0, 1}
    assert aps_set(row, 2.) == {0, 1, 2}


def test_group_thresholds_and_unseen_group_fallback_have_different_observable_sets():
    probabilities, truth = calibration_data()
    calibrator = Conformal(alpha=.5, min_group=2, seed=0).fit(probabilities, truth, ['A', 'A', 'B', 'B'])
    row = np.array([.8, .1, .1])
    assert calibrator.predict_set(row, 'A') == {0}
    assert calibrator.predict_set(row, 'B') == {0, 1, 2}
    assert calibrator.predict_set(row, 'unseen') == calibrator.predict_set(row) == {0, 1, 2}


@pytest.mark.parametrize('groups', [None, ['one', 'two', 'three', 'four'], ['new'] * 4])
def test_refitting_replaces_old_group_thresholds(groups):
    probabilities, truth = calibration_data()
    calibrator = Conformal(min_group=2).fit(probabilities, truth, ['old'] * 4)
    assert 'old' in calibrator.q_group
    calibrator.fit(probabilities, truth, groups)
    assert set(calibrator.q_group) == ({'new'} if groups == ['new'] * 4 else set())
    assert calibrator.predict_set(probabilities[0], 'old') == calibrator.predict_set(probabilities[0])


def test_invalid_refit_keeps_thresholds_and_does_not_advance_the_local_generator():
    probabilities, truth = calibration_data()
    calibrator = Conformal(min_group=2).fit(probabilities, truth, ['old'] * 4)
    before = calibrator.q_pooled, dict(calibrator.q_group), calibrator.class_count
    random_state = deepcopy(calibrator.rng.bit_generator.state)
    with pytest.raises(InputError):
        calibrator.fit(probabilities, truth, ['wrong length'])
    assert (calibrator.q_pooled, calibrator.q_group, calibrator.class_count) == before
    assert calibrator.rng.bit_generator.state == random_state


def test_fitting_is_required_and_class_width_is_bound_to_the_fitted_vocabulary():
    calibrator = Conformal()
    with pytest.raises(ConfigurationError, match='fit calibration'):
        calibrator.predict_set([.8, .1, .1])
    calibrator.fit(*calibration_data())
    with pytest.raises(InputError, match='width'):
        calibrator.predict_set([.5, .5])
    with pytest.raises(InputError, match='hashable'):
        calibrator.predict_set([.8, .1, .1], group=[])


@pytest.mark.parametrize('fields', [{'alpha': 0}, {'alpha': 1}, {'alpha': True}, {'alpha': float('nan')},
    {'alpha': 2 ** 10000}, {'lam': -.1}, {'lam': float('inf')}, {'lam': '0.1'},
    {'k_reg': -1}, {'k_reg': True}, {'k_reg': .5}, {'min_group': 0}, {'seed': -1}])
def test_invalid_fit_configuration_has_parameter_context(fields):
    with pytest.raises(ConfigurationError) as caught:
        Conformal(**fields)
    assert caught.value.context['stage'] == 'conformal_calibration'
    assert caught.value.context['field'] in fields


@pytest.mark.parametrize('row', [[], [[.5, .5]], [.2, .2], [-.1, 1.1], [True, False], ['.5', '.5']])
def test_malformed_probability_rows_fail_before_constructing_a_set(row):
    with pytest.raises(InputError):
        FrozenConformal(.9).predict_set(row)


@pytest.mark.parametrize('row', [[float('nan'), .5], [float('inf'), 0.]])
def test_nonfinite_probabilities_raise_numerical_errors(row):
    with pytest.raises(NumericalError):
        aps_set(row, .9)


@pytest.mark.parametrize('truth', [[0], [0., 0., 1., 1.], [-1, 0, 1, 1], [3, 0, 1, 1],
    [True, False, True, False], [[0], [0], [1], [1]], [[0], [1, 2]]])
def test_truth_is_an_aligned_vector_of_integer_class_ids(truth):
    with pytest.raises(InputError):
        Conformal().fit(calibration_data()[0], truth)


@pytest.mark.parametrize('groups', [['one'], [None] * 4, [float('nan')] * 4,
    [[1], [2], [3], [4]], np.array(['one', 1, 'two', 2], dtype=object)])
def test_invalid_group_annotations_are_rejected(groups):
    with pytest.raises(InputError):
        Conformal().fit(*calibration_data(), groups=groups)


def test_probability_and_score_array_shape_errors_have_domain_errors():
    with pytest.raises(InputError):
        Conformal().fit([[.5, .5], [.2]], [0, 1])
    with pytest.raises(InputError):
        conformal_quantile([], .1)
    with pytest.raises(NumericalError):
        conformal_quantile([float('nan')], .1)
    with pytest.raises(ConfigurationError):
        aps_scores(*calibration_data(), rng=None)


@pytest.mark.parametrize('fields', [{'qhat': -1}, {'qhat': None}, {'qhat': float('nan')}, {'qhat': .9, 'lam': -.1}])
def test_invalid_frozen_configuration_is_rejected(fields):
    with pytest.raises(ConfigurationError):
        FrozenConformal(**fields)


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_supported_precision_is_validated_without_renormalizing_input(dtype):
    row = np.array([.5, .3, .2], dtype=dtype)
    before = row.copy()
    assert FrozenConformal(.6).predict_set(row) == {0, 1}
    np.testing.assert_array_equal(row, before)


def test_prediction_does_not_consume_calibration_randomness():
    calibrator = Conformal().fit(*calibration_data())
    state = deepcopy(calibrator.rng.bit_generator.state)
    calibrator.predict_set([.8, .1, .1])
    assert calibrator.rng.bit_generator.state == state
