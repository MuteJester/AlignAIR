"""Scoring rejects invalid requests before work and restores caller runtime state."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from alignair.train.convergence.evaluator import evaluate_checkpoint
from alignair.errors import AlignAIRError, ConfigurationError, InferenceError, InputError

pytestmark = pytest.mark.component


@pytest.mark.parametrize("defect", ["model", "config", "profile", "model_config", "vocab", "orientation",
    "maps", "missing_map", "map_gene", "map_inverse", "class_count", "meta", "d_missing", "d_width",
    "d_gap", "d_duplicate", "d_bool", "cohort", "strata", "dc_name", "clean_missing", "rows",
    "calibration", "row", "stratum_name", "v_call", "j_call", "d_flag", "d_call", "late_sequence", "long_sequence",
    "source_coordinates", "source_sequence", "source_label"])
def test_invalid_requests_do_not_place_forward_or_change_modes(evaluation_inputs, monkeypatch, defect):
    model, cfg, maps, meta, cohort = evaluation_inputs
    model.train()
    model.branches['v'].eval()
    modes = [child.training for child in model.modules()]
    original = model
    def forbidden(*args, **kwargs):
        pytest.fail("invalid request reached model placement/forward")
    monkeypatch.setattr(model, 'to', forbidden)
    monkeypatch.setattr(model, 'forward', forbidden)
    if defect == 'model': model = None
    elif defect == 'config': cfg = None
    elif defect == 'profile': cfg = replace(cfg, classification_mode='multilabel'); original.cfg = cfg
    elif defect == 'model_config': cfg = replace(cfg, max_seq_length=192)
    elif defect == 'vocab': cfg.vocab_size = 6
    elif defect == 'orientation': cfg.use_orientation = True
    elif defect == 'maps': maps = None
    elif defect == 'missing_map': maps.pop('V')
    elif defect == 'map_gene': maps['V'].gene = 'J'
    elif defect == 'map_inverse': maps['V'].allele_class['V0*01'] = 1
    elif defect == 'class_count': cfg.v_allele_count = 3
    elif defect == 'meta': meta = None
    elif defect == 'd_missing': meta = {}
    elif defect == 'd_width': meta['d_index'].pop('D1*01')
    elif defect == 'd_gap': meta['d_index']['D1*01'] = 3
    elif defect == 'd_duplicate': meta['d_index']['D1*01'] = 0
    elif defect == 'd_bool': meta['d_index']['D1*01'] = True
    elif defect == 'cohort': cohort = None
    elif defect == 'strata': cohort['by_stratum'] = []
    elif defect == 'dc_name': cohort['dc_name'] = None
    elif defect == 'clean_missing': cohort['by_stratum'].pop('clean')
    elif defect == 'rows': cohort['by_stratum']['moderate_full'] = iter([])
    elif defect == 'calibration': cohort['calibration'] = 'bad'
    elif defect == 'row': cohort['calibration'][-1] = None
    elif defect == 'stratum_name': cohort['by_stratum'][''] = []
    elif defect == 'v_call': cohort['calibration'][-1]['v_call'] = 'UNKNOWN'
    elif defect == 'j_call': cohort['by_stratum']['clean'][-1]['j_call'] = 'UNKNOWN'
    elif defect == 'd_flag': cohort['by_stratum']['clean'][-1]['d_inverted'] = 'false'
    elif defect == 'd_call': cohort['by_stratum']['clean'][-1]['d_call'] = 2
    elif defect == 'late_sequence': cohort['calibration'][-1]['aa'] = 'BADZ'
    elif defect == 'long_sequence': cohort['calibration'][-1]['aa'] = 'A' * 129
    elif defect == 'source_coordinates': cohort['by_stratum']['clean'][-1]['record']['v_sequence_end'] = 1000
    elif defect == 'source_sequence': cohort['by_stratum']['clean'][-1]['record']['sequence_aa'] = 'A' * 20
    elif defect == 'source_label': cohort['by_stratum']['clean'][-1]['record']['v_call'] = 'V0*01'
    with pytest.raises(AlignAIRError):
        evaluate_checkpoint(model, cfg, maps, meta, cohort)
    assert [child.training for child in original.modules()] == modes


@pytest.mark.parametrize('options', [{'alpha': 0}, {'alpha': 1}, {'alpha': float('nan')},
    {'freq_edges': [0., .5, .5, 1.]}, {'freq_edges': [0., float('nan'), 1.]},
    {'freq_edges': [-.1, 1.]}, {'freq_edges': [0., 1.1]}, {'freq_edges': [True, False]},
    {'freq_edges': [[0., 1.]]}, {'freq_edges': [[0.], [.5, 1.]]}, {'device': 'invalid-device'}])
def test_invalid_report_options_fail_before_model_placement_even_without_calibration(evaluation_inputs, monkeypatch, options):
    model, cfg, maps, meta, cohort = evaluation_inputs
    cohort['calibration'] = []
    monkeypatch.setattr(model, 'to', lambda *_: pytest.fail('model placed before preflight'))
    with pytest.raises(ConfigurationError):
        evaluate_checkpoint(model, cfg, maps, meta, cohort, **options)


def test_late_input_failure_identifies_stratum_and_source_position(evaluation_inputs):
    model, cfg, maps, meta, cohort = evaluation_inputs
    cohort['calibration'][-1]['aa'] = 'Z'
    with pytest.raises(InputError) as caught:
        evaluate_checkpoint(model, cfg, maps, meta, cohort)
    assert caught.value.context['stratum'] == '__calibration__'
    assert caught.value.context['record_index'] == 1
    assert caught.value.context['stage'] == 'convergence_preflight'


@pytest.mark.parametrize('failure', ['none', 'forward', 'placement', 'width', 'calibration'])
def test_mixed_module_modes_and_gradient_state_are_restored_on_every_exit(evaluation_inputs, monkeypatch, failure):
    from alignair.train.convergence import evaluator
    model, cfg, maps, meta, cohort = evaluation_inputs
    model.train()
    model.branches['v'].eval()
    modes = [child.training for child in model.modules()]
    weights = deepcopy(model.state_dict())
    original_forward = model.forward
    def forward(batch):
        assert not torch.is_grad_enabled() and not any(child.training for child in model.modules())
        if failure == 'forward':
            raise RuntimeError('forward failure')
        result = original_forward(batch)
        if failure == 'width': result['v_class_logits'] = result['v_class_logits'][:, :1]
        return result
    monkeypatch.setattr(model, 'forward', forward)
    def fail(*args):
        raise RuntimeError(f'{failure} failure')
    if failure == 'placement':
        monkeypatch.setattr(model, 'to', fail)
    if failure == 'calibration':
        monkeypatch.setattr(evaluator.Conformal, 'fit', fail)
    if failure == 'none':
        evaluate_checkpoint(model, cfg, maps, meta, cohort)
    else:
        with pytest.raises(InferenceError if failure == 'width' else RuntimeError):
            evaluate_checkpoint(model, cfg, maps, meta, cohort)
    assert torch.is_grad_enabled()
    assert [child.training for child in model.modules()] == modes
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)


def test_inverted_or_unknown_d_truth_is_excluded_without_losing_vj_support(evaluation_inputs):
    model, cfg, maps, meta, cohort = evaluation_inputs
    cohort['by_stratum']['clean'][0]['d_inverted'] = True
    cohort['by_stratum']['clean'][0].pop('d_call')
    cohort['by_stratum']['clean'][1]['d_call'] = 'UNKNOWN'
    result = evaluate_checkpoint(model, cfg, maps, meta, cohort)
    assert result['d_allele_acc'] is result['d_gene_acc'] is None
    assert result['by_stratum']['clean']['n'] == 2


def test_no_d_model_needs_no_d_metadata_or_truth(evaluation_inputs):
    _, cfg, maps, _, cohort = evaluation_inputs
    from alignair.core import AlignAIR
    cfg = replace(cfg, has_d=False, d_allele_count=0)
    for row in cohort['by_stratum']['clean']:
        row.pop('d_call'); row.pop('d_inverted')
    result = evaluate_checkpoint(AlignAIR(cfg), cfg, maps, {}, cohort)
    assert result['d_allele_acc'] is None


@pytest.mark.parametrize('gene,pass_index,stratum', [('v', 1, 'clean'), ('j', 1, 'clean'),
    ('d', 1, 'clean'), ('v', 6, '__calibration__')])
def test_output_width_errors_name_the_head_and_stratum(evaluation_inputs, monkeypatch, gene, pass_index, stratum):
    model, cfg, maps, meta, cohort = evaluation_inputs
    original = model.forward
    calls = 0
    def forward(batch):
        nonlocal calls
        calls += 1
        result = original(batch)
        if calls == pass_index:
            result[f'{gene}_class_logits'] = result[f'{gene}_class_logits'][:, :1]
        return result
    monkeypatch.setattr(model, 'forward', forward)
    with pytest.raises(InferenceError) as caught:
        evaluate_checkpoint(model, cfg, maps, meta, cohort)
    assert caught.value.context['field'] == f'{gene}_class_logits'
    assert caught.value.context['stratum'] == stratum
    assert calls == pass_index


def test_existing_no_grad_context_remains_disabled(evaluation_inputs):
    model, cfg, maps, meta, cohort = evaluation_inputs
    cohort['by_stratum']['empty'] = []
    model.train()
    modes = [child.training for child in model.modules()]
    with torch.no_grad():
        evaluate_checkpoint(model, cfg, maps, meta, cohort)
        assert not torch.is_grad_enabled()
    assert [child.training for child in model.modules()] == modes
