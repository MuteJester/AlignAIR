"""Ordered cohort observations: alphabet independence, support, masks and failure scopes."""
from dataclasses import asdict, replace
import json

import pytest
import torch

from alignair.core import AlignAIRConfig
from alignair.errors import ConfigurationError, NumericalError, TrainingDataError, current_error_context
from alignair.evaluate import collect_alignment_observations

pytestmark = pytest.mark.component


class Echo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = torch.nn.Dropout().eval()
        self.calls = 0

    def forward(self, inputs):
        assert not self.training and not torch.is_grad_enabled()
        self.calls += 1
        return inputs


def fixture(categorical=True):
    cfg = AlignAIRConfig(v_allele_count=2, j_allele_count=2, has_d=False,
                         max_seq_length=16, use_orientation=False, use_meta_heads=False)
    if categorical:
        cfg = cfg.with_amino_acid_heads()
    outputs, targets = {}, {}
    for gene in ('v','j'):
        outputs[cfg.classification_key(gene)] = torch.tensor([[2.,1.], [0.,3.], [2.,1.], [0.,3.], [2.,1.]])
        if categorical:
            targets[f'{gene}_class'] = torch.tensor([0,-1,1,0,1] if gene == 'v' else [1,0,1,0,1])
        else:
            outputs[cfg.classification_key(gene)] = torch.tensor([[.9,.1], [.1,.9], [.9,.1], [.1,.9], [.9,.1]])
            targets[f'{gene}_allele'] = torch.tensor([[1.,1.], [0.,0.], [0.,1.], [1.,0.], [1.,0.]])
        for boundary in ('start','end'):
            outputs[f'{gene}_{boundary}'] = torch.arange(5.).reshape(-1,1) + 1
            targets[f'{gene}_{boundary}'] = torch.arange(5.).reshape(-1,1)
    targets['v_valid'] = torch.tensor([1.,1.,0.,1.,1.])
    targets['v_supervise'] = torch.tensor([True,True,True,True,False])
    targets['v_class_supervise'] = torch.tensor([True,True,True,False,True])

    def adapter(records):
        rows = list(records)
        return ({key:value[rows] for key,value in outputs.items()},
                {key:value[rows] for key,value in targets.items()})

    return cfg, outputs, targets, adapter


@pytest.mark.parametrize('categorical', [True,False])
@pytest.mark.parametrize('protein', [True,False])
def test_one_collector_preserves_order_tail_and_semantics_for_both_input_alphabets(categorical, protein):
    cfg, _, _, adapter = fixture(categorical)
    cfg = replace(cfg, vocab_size=23 if protein else 6)
    model = Echo()
    modes = [module.training for module in model.modules()]
    observed = collect_alignment_observations(model, iter(range(5)), adapter, cfg, batch_size=2)
    assert observed.record_count == 5 and model.calls == 3
    assert [module.training for module in model.modules()] == modes
    assert [row.record_index for row in observed.classification['v']] == ([0] if categorical else [0,3])
    assert [row.record_index for row in observed.boundaries['v_start']] == [0,1,3]
    assert [row.absolute_error for row in observed.boundaries['v_start']] == [1.,1.,1.]
    assert observed.classification['v'][0].truth == ((0,) if categorical else (0,1))
    assert observed.classification['v'][0].correct
    if not categorical:
        assert not observed.classification['v'][1].correct
    assert observed.support['classification']['v'] == (1 if categorical else 2)
    assert observed.support['boundaries']['j_end'] == 5
    json.dumps(asdict(observed), allow_nan=False)  # no tensors, graphs or device buffers retained
    assert current_error_context() == {}


def test_compatible_truth_is_preserved_instead_of_collapsed_to_a_canonical_class():
    cfg, _, targets, adapter = fixture()
    targets['v_compatible'] = torch.tensor([[1.,1.], [0.,1.], [0.,0.], [0.,0.], [0.,0.]])
    observed = collect_alignment_observations(Echo(), range(5), adapter, cfg, batch_size=2)
    assert [row.truth for row in observed.classification['v']] == [(0,1),(1,)]
    assert [row.record_index for row in observed.classification['v']] == [0,1]
    assert all(row.correct for row in observed.classification['v'])
    assert observed.support['boundaries']['v_start'] == 3


@pytest.mark.parametrize('categorical', [True,False])
def test_gene_subsets_do_not_require_unselected_outputs(categorical):
    cfg, outputs, targets, adapter = fixture(categorical)
    cfg = replace(cfg, has_d=True, d_allele_count=2)
    for key in tuple(outputs):
        if key.startswith('j_'):
            del outputs[key]
    result = collect_alignment_observations(Echo(), range(5), adapter, cfg, genes=('v',))
    assert result.classification.keys() == {'v'}
    assert result.boundaries.keys() == {'v_start','v_end'}


def test_empty_cohort_preserves_configured_support_without_adapting_or_predicting():
    cfg, _, _, _ = fixture()
    model = Echo()
    result = collect_alignment_observations(model, [], lambda _: pytest.fail('adapted empty cohort'), cfg)
    assert result.record_count == model.calls == 0
    assert result.support['classification'] == {'v':0,'j':0}
    assert result.support['boundaries']['v_start'] == 0


@pytest.mark.parametrize('size', [True,False,0,-1,1.5,None])
def test_invalid_batch_size_fails_before_source_iteration_or_mode_changes(size):
    cfg, _, _, adapter = fixture()
    class Unopened:
        def __iter__(self):
            pytest.fail('iterated source during preflight')
    model = Echo()
    with pytest.raises(ConfigurationError) as caught:
        collect_alignment_observations(model, Unopened(), adapter, cfg, batch_size=size)
    assert caught.value.context['field'] == 'batch_size'
    assert model.training and not model.child.training and model.calls == 0


@pytest.mark.parametrize('genes', [(), ('x',), ('v','v'), 'vj', 1, ([],)])
def test_invalid_gene_selection_is_rejected(genes):
    cfg, _, _, adapter = fixture()
    with pytest.raises(ConfigurationError, match='genes'):
        collect_alignment_observations(Echo(), [], adapter, cfg, genes=genes)


@pytest.mark.parametrize('failure', ['model','config','adapter','class_count'])
def test_invalid_assembly_is_rejected_before_collection(failure):
    cfg, _, _, adapter = fixture()
    model = Echo()
    if failure == 'model': model = object()
    elif failure == 'config': cfg = None
    elif failure == 'adapter': adapter = None
    else: cfg = replace(cfg, v_allele_count=0)
    with pytest.raises(ConfigurationError):
        collect_alignment_observations(model, [], adapter, cfg)


@pytest.mark.parametrize('failure', ['scores_rows','scores_classes','scores_missing','scores_integer',
                                    'class_shape','class_dtype','class_range','boundary_shape','boundary_missing',
                                    'boundary_complex','boundary_boolean','wrong_device','class_mask'])
def test_invalid_fields_have_batch_gene_and_field_context_and_restore_modes(failure):
    cfg, outputs, targets, adapter = fixture()
    if failure == 'scores_rows': outputs['v_class_logits'] = torch.ones(4,2)
    elif failure == 'scores_classes': outputs['v_class_logits'] = torch.ones(5,3)
    elif failure == 'scores_missing': del outputs['v_class_logits']
    elif failure == 'scores_integer': outputs['v_class_logits'] = outputs['v_class_logits'].long()
    elif failure == 'class_shape': targets['v_class'] = targets['v_class'][:,None]
    elif failure == 'class_dtype': targets['v_class'] = targets['v_class'].float()
    elif failure == 'class_range': targets['v_class'][0] = 2
    elif failure == 'boundary_shape': outputs['v_start'] = torch.zeros(5,2)
    elif failure == 'boundary_missing': del targets['v_start']
    elif failure == 'boundary_complex': outputs['v_start'] = outputs['v_start'].to(torch.complex64)
    elif failure == 'boundary_boolean': targets['v_start'] = targets['v_start'].bool()
    elif failure == 'wrong_device': targets['v_start'] = targets['v_start'].to('meta')
    else: targets['v_class_supervise'][0] = False; targets['v_class_supervise'] = torch.full((5,),.5)
    model = Echo()
    # Use one chunk so malformed fixture row counts reach the collector unchanged.
    with pytest.raises(TrainingDataError) as caught:
        collect_alignment_observations(model, range(5), lambda _: (outputs,targets), cfg)
    assert caught.value.context['stage'] == 'cohort_collection'
    assert caught.value.context['record_offset'] == 0
    assert caught.value.context['gene'] == 'v'
    assert 'field' in caught.value.context
    assert model.training and not model.child.training
    assert current_error_context() == {}


@pytest.mark.parametrize('failure', ['score','predicted_boundary','target_boundary','error_overflow'])
def test_nonfinite_supervised_values_are_numerical_errors(failure):
    cfg, outputs, targets, adapter = fixture()
    if failure == 'score': outputs['v_class_logits'][0,0] = float('nan')
    elif failure == 'predicted_boundary': outputs['v_start'][0,0] = float('inf')
    elif failure == 'target_boundary': targets['v_start'][0,0] = float('nan')
    else:
        outputs['v_start'][0,0] = 3e38
        targets['v_start'][0,0] = -3e38
    with pytest.raises(NumericalError) as caught:
        collect_alignment_observations(Echo(), range(5), adapter, cfg)
    assert caught.value.context['field'].startswith('v_')
    assert current_error_context() == {}


def test_masked_placeholders_do_not_pollute_observed_results():
    cfg, outputs, targets, adapter = fixture()
    outputs['v_class_logits'][2] = float('nan')
    outputs['v_start'][2] = float('nan')
    targets['v_start'][2] = float('nan')
    result = collect_alignment_observations(Echo(), range(5), adapter, cfg)
    assert result.support['classification']['v'] == 1
    assert [row.record_index for row in result.boundaries['v_start']] == [0,1,3]


@pytest.mark.parametrize('failure', ['truth','score'])
def test_multilabel_membership_and_probability_contracts_are_checked(failure):
    cfg, outputs, targets, adapter = fixture(False)
    if failure == 'truth': targets['v_allele'][0,0] = .5
    else: outputs['v_allele'][0,0] = 2.
    with pytest.raises(TrainingDataError if failure == 'truth' else NumericalError):
        collect_alignment_observations(Echo(), range(5), adapter, cfg)


@pytest.mark.parametrize('failure', ['batch_arity','inputs','targets','outputs','source','adapter','model'])
def test_boundary_failures_restore_modes_and_do_not_leak_context(failure):
    cfg, _, _, adapter = fixture()
    model = Echo()
    records = range(5)
    def broken(*_):
        raise RuntimeError('injected failure')
    if failure == 'batch_arity': adapter = lambda _: ()
    elif failure == 'inputs': adapter = lambda _: (None,{})
    elif failure == 'targets': adapter = lambda _: ({},None)
    elif failure == 'outputs': model.forward = lambda _: None
    elif failure == 'source':
        def records_iter():
            yield 0
            raise RuntimeError('injected failure')
        records = records_iter()
    elif failure == 'adapter': adapter = broken
    else: model.forward = broken
    with pytest.raises(TrainingDataError if failure in ('batch_arity','inputs','targets','outputs') else RuntimeError):
        collect_alignment_observations(model, records, adapter, cfg)
    assert model.training and not model.child.training
    assert current_error_context() == {}
