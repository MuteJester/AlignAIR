"""Protein-study inference/strata composed with shared cohort report arithmetic."""
from types import SimpleNamespace

import pytest
import torch

from alignair.aa.evaluate import evaluate_frozen
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


@pytest.mark.parametrize('batch_size', [1,2,3,8])
def test_frozen_report_includes_tail_and_preserves_study_strata_and_module_modes(batch_size):
    model = CohortModel()
    modes = [module.training for module in model.modules()]
    report = evaluate_frozen(model, cohort(), config(), maps(), batch_size=batch_size)
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
    report = evaluate_frozen(model, [], config(), maps())
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
        evaluate_frozen(model, cohort(), config(), maps())
    assert [module.training for module in model.modules()] == modes


def test_frozen_report_omits_unknown_classes_while_retaining_observed_boundaries():
    model = CohortModel()
    samples = cohort()
    samples[0]['v_class'] = -1
    report = evaluate_frozen(model, samples, config(), maps())
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
    report = evaluate_frozen(CohortModel(), samples, config(), maps(), batch_size=2)
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
        evaluate_frozen(CohortModel(), samples, config(), maps())


def test_categorical_study_rejects_a_multilabel_configuration():
    cfg = AlignAIRConfig(v_allele_count=2, j_allele_count=2, has_d=False)
    with pytest.raises(ConfigurationError, match='categorical'):
        evaluate_frozen(CohortModel(), cohort(), cfg, maps())


def test_study_rejects_an_unfrozen_iterator_before_collecting_observations():
    model = CohortModel()
    with pytest.raises(ConfigurationError, match='indexable'):
        evaluate_frozen(model, iter(cohort()), config(), maps())
    assert model.calls == 0
