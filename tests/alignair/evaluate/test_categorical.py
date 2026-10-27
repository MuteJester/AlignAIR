"""Exact-class evaluation, explicit study strata and independent observed support."""
from types import SimpleNamespace

import pytest
import torch

from alignair.data.adapters import CategoricalBatchAdapter
from alignair.evaluate import ClassificationStratum, evaluate_categorical_cohort, protein_v_strata
from alignair.errors import ConfigurationError
from alignair.core import AlignAIRConfig

pytestmark = pytest.mark.integration


class CohortModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = torch.nn.Dropout().eval()
        self.calls = 0

    def forward(self, inputs):
        assert not self.training and not torch.is_grad_enabled()
        self.calls += 1
        rows = inputs['tokenized_sequence'][:,0] - 1
        labels = torch.tensor([0,1,1])[rows]
        return {
            'v_class_logits': torch.nn.functional.one_hot(labels,2).float(),
            'j_class_logits': torch.tensor([[0.,1.]]).repeat(len(rows),1),
            'v_start': torch.tensor([1.,2.,3.])[rows,None],
            'v_end': torch.tensor([21.,61.,91.])[rows,None],
            'j_start': torch.full((len(rows),1),89.),
            'j_end': torch.full((len(rows),1),100.),
        }


def cohort():
    return [{'tokens': [i+1]*100, 'v_class': v, 'j_class': j,
             'v_start': 0., 'v_end': end, 'j_start': 90., 'j_end': 100.}
            for i,(v,j,end) in enumerate([(0,0,20.), (0,1,60.), (1,1,90.)])]


def config():
    return AlignAIRConfig(v_allele_count=2, j_allele_count=2, has_d=False,
                          max_seq_length=192).with_amino_acid_heads()


def maps():
    return {'V': SimpleNamespace(ambiguity_set=lambda label: [0] if label == 0 else [1,2,3])}


def report_options():
    cfg = config()
    return {'batch_adapter': CategoricalBatchAdapter.from_config(cfg), 'config': cfg,
            'genes': ('v', 'j'), 'strata': protein_v_strata(maps()['V'])}


@pytest.mark.parametrize('batch_size', [1,2,3,8])
def test_frozen_report_includes_tail_and_preserves_study_strata_and_module_modes(batch_size):
    model = CohortModel()
    modes = [module.training for module in model.modules()]
    report = evaluate_categorical_cohort(model, cohort(), **report_options(), batch_size=batch_size)
    assert model.calls == (3+batch_size-1)//batch_size
    assert [module.training for module in model.modules()] == modes
    assert report['n'] == 3
    assert report['classification']['v']['micro_acc'] == pytest.approx(2/3)
    assert report['classification']['v']['macro_acc'] == .75
    assert report['classification']['v']['majority_baseline'] == pytest.approx(2/3)
    assert report['boundaries']['v_start']['mae'] == 2.
    assert report['boundaries']['v_end']['constant_median_baseline_mae'] == pytest.approx(70/3)
    assert report['boundaries']['j_start']['mae_ci95'] == (1.,1.)
    assert report['boundaries']['j_end']['mae'] == 0.
    coverage = report['stratified']['v_acc_by_coverage']
    assert {k:(v['acc'],v['n']) for k,v in coverage.items()} == {'<40':(1.,1), '40-79':(0.,1), '>=80':(1.,1)}
    ambiguity = report['stratified']['v_acc_by_ambiguity_size']
    assert {k:(v['acc'],v['n']) for k,v in ambiguity.items()} == {'1':(.5,2), '>=3':(1.,1)}


def test_empty_frozen_cohort_does_not_run_model_or_invent_strata():
    model = CohortModel()
    report = evaluate_categorical_cohort(model, [], **report_options())
    assert report['n'] == model.calls == 0
    assert report['stratified'] == {'v_acc_by_coverage':{}, 'v_acc_by_ambiguity_size':{}}
    assert report['classification']['v']['paired_lift']['mean'] == 0.


def test_frozen_inference_failure_restores_each_modules_mode():
    class Broken(CohortModel):
        def forward(self, inputs):
            raise RuntimeError('failed cohort inference')

    model = Broken()
    modes = [module.training for module in model.modules()]
    with pytest.raises(RuntimeError, match='cohort inference'):
        evaluate_categorical_cohort(model, cohort(), **report_options())
    assert [module.training for module in model.modules()] == modes


def test_frozen_report_omits_unknown_classes_while_retaining_observed_boundaries():
    model = CohortModel()
    samples = cohort()
    samples[0]['v_class'] = -1
    report = evaluate_categorical_cohort(model, samples, **report_options())
    assert report['n'] == 3
    assert report['support']['classification']['v'] == 2
    assert report['support']['boundaries']['v_start'] == 3
    assert report['classification']['v']['micro_acc'] == .5
    assert '<40' not in report['stratified']['v_acc_by_coverage']
    assert model.training and not model.child.training


@pytest.mark.parametrize('mask', ['v_valid', 'v_supervise', 'v_class_supervise'])
def test_supervision_filtering_keeps_original_record_indices_for_strata(mask):
    samples = cohort()
    samples[0][mask] = False
    report = evaluate_categorical_cohort(CohortModel(), samples, **report_options(), batch_size=2)
    assert report['support']['classification']['v'] == 2
    assert report['support']['boundaries']['v_start'] == (3 if mask == 'v_class_supervise' else 2)
    assert report['support']['classification']['j'] == 3
    assert report['stratified']['v_acc_by_coverage'].keys() == {'40-79', '>=80'}
    assert report['stratified']['v_acc_by_coverage']['40-79']['acc'] == 0.
    assert report['stratified']['v_acc_by_ambiguity_size']['1']['acc'] == 0.


def test_exact_class_study_refuses_to_guess_one_label_from_compatible_truth():
    samples = cohort()
    for sample in samples:
        sample['v_compatible'] = {0,1}
    with pytest.raises(ConfigurationError, match='singleton truth'):
        evaluate_categorical_cohort(CohortModel(), samples, **report_options())


def test_categorical_study_rejects_a_multilabel_configuration():
    cfg = AlignAIRConfig(v_allele_count=2, j_allele_count=2, has_d=False)
    with pytest.raises(ConfigurationError, match='categorical'):
        evaluate_categorical_cohort(CohortModel(), cohort(), lambda records: None, cfg)


def test_study_rejects_an_unfrozen_iterator_before_collecting_observations():
    model = CohortModel()
    with pytest.raises(ConfigurationError, match='indexable'):
        evaluate_categorical_cohort(model, iter(cohort()), **report_options())
    assert model.calls == 0


@pytest.mark.parametrize('field,value', [('name', ''), ('name', True), ('gene', None), ('gene', ' '), ('group', 0)])
def test_invalid_stratum_definitions_are_configuration_errors(field, value):
    fields = {'name': 'coverage', 'gene': 'v', 'group': lambda record, truth: 'observed'}
    with pytest.raises(ConfigurationError):
        ClassificationStratum(**{**fields, field: value})


@pytest.mark.parametrize('strata', [None, ['bad'],
    [ClassificationStratum('same', 'v', str), ClassificationStratum('same', 'j', str)],
    [ClassificationStratum('d', 'd', str)]])
def test_invalid_strata_fail_before_consuming_records(strata):
    model = CohortModel()
    class Untouched:
        def __iter__(self):
            pytest.fail('invalid request consumed the source')
    options = {**report_options(), 'strata': strata}
    with pytest.raises(ConfigurationError):
        evaluate_categorical_cohort(model, Untouched(), **options)
    assert model.calls == 0


@pytest.mark.parametrize('label', [None, '', ' ', 1])
def test_invalid_group_labels_have_stratum_and_original_record_context(label):
    model = CohortModel()
    options = {**report_options(), 'strata': (ClassificationStratum('custom', 'v', lambda record, truth: label),)}
    with pytest.raises(ConfigurationError, match='labels must be nonempty') as caught:
        evaluate_categorical_cohort(model, cohort(), **options)
    assert caught.value.context['stage'] == 'classification_stratification'
    assert caught.value.context['stratum'] == 'custom'
    assert caught.value.context['record_index'] == 0
    assert model.training and not model.child.training


def test_custom_strata_join_only_supervised_original_rows_and_singleton_truth():
    samples = cohort()
    samples[0]['v_class_supervise'] = False
    received = []
    def group(record, truth):
        received.append((record, truth))
        return 'second' if record is samples[1] else 'third'
    options = {**report_options(), 'strata': (ClassificationStratum('selected', 'v', group),)}
    report = evaluate_categorical_cohort(CohortModel(), samples, **options, batch_size=2)
    assert received == [(samples[1], 0), (samples[2], 1)]
    assert report['stratified']['selected']['second']['acc'] == 0.
    assert report['stratified']['selected']['third']['acc'] == 1.
    assert report['support']['boundaries']['v_start'] == 3


def test_no_strata_supports_an_iterator_and_gene_selection_iterables():
    options = {**report_options(), 'strata': (), 'genes': iter(('j',))}
    report = evaluate_categorical_cohort(CohortModel(), iter(cohort()), **options, batch_size=2)
    assert list(report['classification']) == ['j']
    assert set(report['boundaries']) == {'j_start', 'j_end'}
    assert report['n'] == 3 and report['stratified'] == {}


def test_protein_preset_requires_an_explicit_class_map():
    with pytest.raises(ConfigurationError, match='class map'):
        protein_v_strata(None)


@pytest.mark.parametrize('span,expected', [(39, '<40'), (40, '40-79'), (79, '40-79'), (80, '>=80')])
def test_established_protein_bins_keep_thresholds_and_two_allele_ambiguity(span, expected):
    samples = [cohort()[0]]
    samples[0]['v_end'] = span
    class_map = SimpleNamespace(ambiguity_set=lambda label: ['one', 'two'])
    options = {**report_options(), 'strata': protein_v_strata(class_map)}
    report = evaluate_categorical_cohort(CohortModel(), samples, **options)
    assert list(report['stratified']['v_acc_by_coverage']) == [expected]
    assert report['stratified']['v_acc_by_ambiguity_size']['2']['n'] == 1
