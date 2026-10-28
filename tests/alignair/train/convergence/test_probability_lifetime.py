"""Stratum reduction releases unused probability buffers without more inference."""
import weakref

import pytest

from alignair.train.convergence import evaluator

pytestmark = pytest.mark.component


@pytest.mark.parametrize('clean_position', ['first', 'middle', 'last'])
def test_only_clean_probability_arrays_survive_between_inference_calls(evaluation_inputs, monkeypatch, clean_position):
    model, cfg, maps, meta, cohort = evaluation_inputs
    strata = cohort['by_stratum']
    others = [name for name in strata if name != 'clean']
    orders = {'first': ['clean', *others], 'middle': [others[0], 'clean', *others[1:]],
              'last': [*others, 'clean']}
    cohort['by_stratum'] = {name: strata[name] for name in orders[clean_position]}
    cohort['by_stratum']['empty'] = []
    calibration_sequences = tuple(row['aa'] for row in cohort['calibration'])
    original = evaluator.protein_class_probabilities
    nonclean_refs, calibration_refs, requests = [], [], []
    clean_refs = {}
    def observed(model, sequences, config, device, keys):
        # Run the actual shared probability path; observe its arrays without holding them alive.
        assert all(reference() is None for reference in nonclean_refs)
        if clean_refs:
            assert clean_refs['v_class_logits']() is not None
        result = original(model, sequences, config, device, keys)
        requests.append(tuple(keys))
        if 'j_class_logits' in keys:
            clean_refs.update({key: weakref.ref(value) for key, value in result.items()})
        elif tuple(sequences) == calibration_sequences:
            calibration_refs.extend(weakref.ref(value) for value in result.values())
        else:
            nonclean_refs.extend(weakref.ref(value) for value in result.values())
        return result
    monkeypatch.setattr(evaluator, 'protein_class_probabilities', observed)
    forward_calls = []
    handle = model.register_forward_hook(lambda *_: forward_calls.append(1))
    try:
        report = evaluator.evaluate_checkpoint(model, cfg, maps, meta, cohort)
    finally:
        handle.remove()
    assert len(nonclean_refs) == len(others) and len(clean_refs) == 3 and len(calibration_refs) == 1
    assert all(reference() is None for reference in nonclean_refs + list(clean_refs.values()) + calibration_refs)
    assert requests.count(('v_class_logits', 'j_class_logits', 'd_class_logits')) == 1
    assert len(requests) == 5 and len(forward_calls) == 6
    assert list(report['by_stratum']) == orders[clean_position]
    assert report['conformal']['n_test'] == len(strata['clean'])
