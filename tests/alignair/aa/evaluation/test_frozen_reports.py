"""Protein-study inference/strata composed with shared cohort report arithmetic."""
from types import SimpleNamespace

import pytest
import torch

from alignair.aa.evaluate import evaluate_frozen
from alignair.errors import InputError

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


def maps():
    return {'V': SimpleNamespace(ambiguity_set=lambda label: [0] if label == 0 else [1,2,3])}


@pytest.mark.parametrize('batch_size', [1,2,3,8])
def test_frozen_report_includes_tail_and_preserves_study_strata_and_module_modes(batch_size):
    model = CohortModel()
    modes = [module.training for module in model.modules()]
    report = evaluate_frozen(model, cohort(), SimpleNamespace(max_seq_length=192), maps(), batch_size=batch_size)
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
    report = evaluate_frozen(model, [], SimpleNamespace(max_seq_length=192), maps())
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
        evaluate_frozen(model, cohort(), SimpleNamespace(max_seq_length=192), maps())
    assert [module.training for module in model.modules()] == modes


def test_frozen_report_rejects_unknown_truth_instead_of_scoring_it_as_wrong():
    model = CohortModel()
    samples = cohort()
    samples[0]['v_class'] = -1
    with pytest.raises(InputError):
        evaluate_frozen(model, samples, SimpleNamespace(max_seq_length=192), maps())
    assert model.training and not model.child.training
