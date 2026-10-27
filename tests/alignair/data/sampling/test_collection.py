"""Budget, consumption, callback failure, and scientific census invariants."""
import pytest
from alignair.data.sampling import SampleResult, SampleBudgetExhausted, collect_samples, check_collection_counts
from alignair.errors import ConfigurationError, TrainingDataError

pytestmark = pytest.mark.component


def _builder(record):
    return SampleResult(None, 'bad') if record < 0 else SampleResult({'tokens': [record]})


def test_collection_separates_rejections_from_selection_without_consuming_an_extra_record():
    source = iter([-1, 0, 1, 2, 3, 4])
    counts = {}
    result = collect_samples(source, _builder, want=2, budget=6, counter=counts,
                             select=lambda sample, _: sample['tokens'][0] % 2 == 1, rejection_reasons=('bad',))
    assert result.samples == [{'tokens': [1]}, {'tokens': [3]}]
    assert result.generated == 5 and next(source) == 4
    assert counts == {'generated': 5, 'bad': 1, 'usable': 4, 'selected': 2, 'selection_skipped': 2}
    check_collection_counts(counts, ('bad',))


@pytest.mark.parametrize('source,budget,generated', [([0, 1, 2, 3], 2, 2), ([0], 5, 1)])
def test_exhaustion_reports_actual_source_consumption(source, budget, generated):
    counts = {}
    with pytest.raises(SampleBudgetExhausted) as caught:
        collect_samples(iter(source), _builder, want=3, budget=budget, counter=counts, rejection_reasons=('bad',))
    assert caught.value.context['generated'] == generated
    assert caught.value.context['budget'] == budget
    assert caught.value.context['collected'] == generated
    check_collection_counts(counts, ('bad',))


@pytest.mark.parametrize('outcome', [SampleResult(None), SampleResult(None, 'unregistered'),
                                    SampleResult({}, 'bad'), SampleResult(3)])
def test_invalid_builder_results_fail_and_remain_accounted_as_errors(outcome):
    counts = {}
    with pytest.raises(TrainingDataError):
        collect_samples([{}], lambda _: outcome, want=1, budget=1, counter=counts, rejection_reasons=('bad',))
    assert counts == {'generated': 1, 'preparation_errors': 1}
    check_collection_counts(counts, ('bad',))


@pytest.mark.parametrize('stage', ['preparation', 'selection'])
def test_callback_defects_propagate_without_becoming_scientific_rejections(stage):
    original = RuntimeError('callback defect')
    def broken(*_): raise original
    counts = {}
    with pytest.raises(RuntimeError) as caught:
        collect_samples([1], broken if stage=='preparation' else _builder, want=1, budget=1,
                        counter=counts, select=broken if stage=='selection' else None)
    assert caught.value is original
    assert counts[stage+'_errors'] == 1
    check_collection_counts(counts)


def test_appending_to_historical_counts_keeps_unknown_selection_outcomes_explicit():
    counts = {'generated': 2, 'usable': 1, 'bad': 1}
    check_collection_counts(counts, ('bad',))
    collect_samples([3], _builder, want=1, budget=1, counter=counts, rejection_reasons=('bad',))
    assert counts == {'generated': 3, 'usable': 2, 'bad': 1, 'selection_untracked': 1, 'selected': 1}
    check_collection_counts(counts, ('bad',))


@pytest.mark.parametrize('change', [dict(want=0), dict(want=True), dict(budget=-1), dict(budget=1.5),
                                   dict(builder=None), dict(select=1), dict(counter=[]),
                                   dict(rejection_reasons='bad'), dict(rejection_reasons=None),
                                   dict(rejection_reasons=('bad','bad')), dict(rejection_reasons=('generated',)),
                                   dict(rejection_reasons=('',)), dict(rejection_reasons=(3,))])
def test_invalid_collection_configuration_does_not_consume_source(change):
    seen = []
    def source():
        seen.append(True)
        yield 1
    kwargs = dict(builder=_builder, want=1, budget=1)
    kwargs.update(change)
    with pytest.raises(ConfigurationError): collect_samples(source(), **kwargs)
    assert seen == []


@pytest.mark.parametrize('counts', [[], {'generated': -1}, {'generated': True}, {'generated': 1},
                                   {'generated': 1, 'usable': 1, 'selected': 2}])
def test_invalid_ledgers_cannot_pass_balancing_checks(counts):
    with pytest.raises(TrainingDataError): check_collection_counts(counts)


def test_counter_validation_cannot_reinterpret_selection_totals_as_rejections():
    with pytest.raises(ConfigurationError):
        check_collection_counts({'generated': 2, 'usable': 1, 'selected': 1}, ('selected',))
