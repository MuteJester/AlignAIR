"""Cohort report arithmetic checked against hand-counted labels and paired errors."""
import importlib

import pytest

from alignair.errors import InputError
from alignair.evaluate import (boundary_error_report, classification_report, constant_median_mae,
                               macro_accuracy, majority_class_baseline)

pytestmark = pytest.mark.component


def test_class_report_uses_observed_class_denominators_and_paired_majority_errors():
    # Class 0: 2/3 right, class 1: 1/1 right -> micro 3/4, macro 5/6.
    report = classification_report([0,0,0,1], [0,0,1,1])
    assert report['micro_acc'] == .75
    assert report['macro_acc'] == pytest.approx(5/6)
    assert report['majority_baseline'] == .75
    assert report['macro_over_micro'] == pytest.approx(10/9)
    assert report['n_distinct_true_classes'] == 2
    # Advantages [0,0,-1,1], not a comparison of independent marginal intervals.
    assert report['paired_lift']['mean'] == 0.
    assert not report['paired_lift']['lower_gt_zero']


def test_majority_tie_follows_first_observed_class():
    report = classification_report([2,2,1,1], [2,2,0,0])
    # Choosing first class 2 gives zero paired advantages on every read.
    assert report['paired_lift'] == {'mean': 0., 'ci95': (0.,0.), 'lower_gt_zero': False}


def test_macro_accuracy_downweights_common_classes():
    assert macro_accuracy([0]*8 + [1]*2, [0]*10) == .5
    assert majority_class_baseline([1,1,1,2]) == .75
    assert majority_class_baseline([]) == 0.
    assert macro_accuracy([], []) == 0.


def test_boundary_report_matches_constant_and_model_errors_per_observation():
    # Upper median = 3, constant errors [2,1,0,7], model errors [1,1,0,3].
    report = boundary_error_report([1.,2.,3.,10.], [1.,1.,0.,3.])
    assert report['mae'] == 1.25
    assert report['constant_median_baseline_mae'] == 2.5
    assert report['relative_reduction'] == .5
    assert report['paired_improvement']['mean'] == 1.25
    assert report == boundary_error_report([1.,2.,3.,10.], [1.,1.,0.,3.], max_resample_elements=1)


def test_constant_median_mae_beats_or_matches_other_constants():
    values = [1.,2.,3.,10.]
    baseline = constant_median_mae(values)
    for constant in (0.,1.5,4.,11.):
        assert baseline <= sum(abs(v-constant) for v in values)/len(values)
    assert constant_median_mae([]) == 0.


def test_empty_and_zero_baselines_retain_legacy_report_fields():
    classification = classification_report([], [])
    assert classification['micro_acc'] == classification['macro_over_micro'] == 0.
    assert classification['micro_acc_ci95'] == (0.,0.)
    boundaries = boundary_error_report([], [])
    assert boundaries['mae'] == boundaries['relative_reduction'] == 0.
    assert boundaries['paired_improvement']['mean'] == 0.
    assert boundary_error_report([2.,2.], [1.,1.])['relative_reduction'] == 0.


@pytest.mark.parametrize("ids", [[-1], [True], [1.5], ['1'], [[1]], [[1],[1,2]], None])
def test_unobserved_or_malformed_class_truth_cannot_be_reported_as_accuracy(ids):
    with pytest.raises(InputError):
        classification_report(ids, [0])


def test_malformed_predictions_and_mismatched_lengths_are_rejected():
    with pytest.raises(InputError):
        classification_report([0], [-1])
    with pytest.raises(InputError):
        macro_accuracy([0,1], [0])


@pytest.mark.parametrize("targets, errors", [([0.], []), ([0.], [-1.]),
                                            ([float('nan')], [0.]), ([0.], [float('inf')])])
def test_boundary_reports_reject_misaligned_nonfinite_or_negative_errors(targets, errors):
    with pytest.raises(InputError):
        boundary_error_report(targets, errors)


def test_convergence_macro_accuracy_uses_shared_report_owner():
    from alignair.aa.convergence import evaluator
    assert evaluator.macro_accuracy is macro_accuracy


def test_public_benchmark_exports_remain_available_and_unknown_attributes_fail():
    evaluation = importlib.import_module('alignair.evaluate')
    benchmark = importlib.import_module('alignair.evaluate.benchmark')
    for name in ('default_strata','format_text','generate_labeled','run_benchmark','score'):
        assert getattr(evaluation, name) is getattr(benchmark, name)
    with pytest.raises(AttributeError):
        getattr(evaluation, 'missing_api')
