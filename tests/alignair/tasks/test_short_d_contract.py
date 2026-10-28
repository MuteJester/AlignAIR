"""The legacy Short-D objective requires a sentinel class, never an arbitrary final D allele."""
from dataclasses import asdict, replace
import json

import pytest
import torch

from alignair.core import AlignAIR, AlignAIRConfig, ModelSetup, prepare_model
from alignair.data.training import prepare_training_data
from alignair.errors import ConfigurationError
from alignair.model_file import container, load_model, read_model_state, serialize
from alignair.reference import GeneReference, ReferenceSet
from alignair.tasks.losses import hierarchical_loss, make_loss_weights
from alignair.train.checkpoints import ResumeState, validate_continuation

pytestmark = pytest.mark.component


def reference(*, sentinel=False):
    panels = {'V': (['IGHV1*01'], ['ATGTGT']),
              'J': (['IGHJ1*01'], ['TGGAAA']),
              'D': (['IGHD1*01', 'Short-D' if sentinel else 'IGHD2*01'], ['GGG', 'NNN'])}
    return ReferenceSet({gene: GeneReference(names, sequences, {name: i for i, name in enumerate(names)})
                         for gene, (names, sequences) in panels.items()}, has_d=True)


@pytest.mark.parametrize('space', ['nucleotide', 'protein'])
@pytest.mark.parametrize('sentinel', [False, True])
def test_reference_bound_objectives_follow_explicit_d_label_identity(space, sentinel):
    source = reference(sentinel=sentinel)
    setup = prepare_model(source, input_space=space, max_seq_length=128, embed_dim=4, filters=4, block_out=8)
    assert setup.config.short_d_penalty is sentinel
    assert next(spec for spec in setup.config.gene_specs if spec.name == 'd').short_d_penalty is sentinel
    assert AlignAIRConfig.from_reference(source).short_d_penalty is sentinel
    assert prepare_training_data(setup).batch_adapter is not None


@pytest.mark.parametrize('value', [None, 0, 1, 'true'])
def test_short_d_objective_setting_is_a_boolean(value):
    with pytest.raises(ConfigurationError) as caught:
        AlignAIRConfig(2, 2, short_d_penalty=value)
    assert caught.value.context['field'] == 'short_d_penalty'


def test_manual_legacy_loss_is_rejected_for_reference_bound_training_without_sentinel():
    setup = prepare_model(reference(), short_d_penalty=True)
    with pytest.raises(ConfigurationError, match='final D label') as caught:
        prepare_training_data(setup)
    assert caught.value.context['field'] == 'short_d_penalty'
    assert 'warm start' in caught.value.hint


@pytest.mark.parametrize('supervised,end,expected', [(True, 3., .8), (True, 7., 0.), (False, 3., 0.)])
def test_disabling_the_penalty_changes_only_the_supervised_short_d_class_term(supervised, end, expected):
    legacy = AlignAIRConfig(2, 2, 2, has_d=True, max_seq_length=8, use_meta_heads=False,
                            use_orientation=False, short_d_penalty=True)
    outputs, targets = {'position_mask': torch.ones(1, 8, dtype=torch.bool)}, {}
    for gene in ('v', 'd', 'j'):
        outputs[f'{gene}_allele'] = torch.tensor([[.2, .8]], requires_grad=True)
        targets[f'{gene}_allele'] = torch.tensor([[1., 0.]])
        for boundary, value in (('start', 1.), ('end', end)):
            outputs[f'{gene}_{boundary}'] = torch.tensor([[value]])
            outputs[f'{gene}_{boundary}_logits'] = torch.zeros(1, 8)
            targets[f'{gene}_{boundary}'] = torch.tensor([[value]])
    targets['d_supervise'] = torch.tensor([supervised])
    weights = make_loss_weights(legacy)
    _, before = hierarchical_loss(outputs, targets, legacy, weights)
    _, after = hierarchical_loss(outputs, targets, replace(legacy, short_d_penalty=False), weights)
    assert (before['classification'] - after['classification']).item() == pytest.approx(expected)
    torch.testing.assert_close(before['segmentation'], after['segmentation'])
    old_gradient = torch.autograd.grad(before['classification'], outputs['d_allele'])[0]
    new_gradient = torch.autograd.grad(after['classification'], outputs['d_allele'])[0]
    torch.testing.assert_close(old_gradient - new_gradient,
                               torch.tensor([[0., 1. if expected else 0.]]))


def test_historical_config_preserves_inference_but_cannot_silently_change_resume_objective(tmp_path):
    source = reference()
    corrected = prepare_model(source, max_seq_length=128, embed_dim=4, filters=4, block_out=8)
    old_config = replace(corrected.config, short_d_penalty=True)
    old_model = AlignAIR(old_config).eval()
    encoded = asdict(old_config)
    encoded.pop('short_d_penalty')  # A historical artifact has no explicit flag.
    path = tmp_path / 'historical.alignair'
    container.write_container(path, {'model': {'label_schema': corrected.label_schema()}}, {
        'config': (json.dumps(encoded).encode(), 'zlib'),
        'weights': (serialize.state_dict_to_bytes(old_model.state_dict()), 'none'),
        'reference_json': (serialize.reference_to_json(source), 'zlib'),
    })
    decoded, restored = read_model_state(path), load_model(path).model
    assert decoded.config.short_d_penalty is restored.cfg.short_d_penalty is True
    tokens = torch.zeros(1, 128, dtype=torch.long)
    tokens[:, :8] = 1
    with torch.no_grad():
        before, after = old_model({'tokenized_sequence': tokens}), restored({'tokenized_sequence': tokens})
    assert before.keys() == after.keys()
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    # Inference restoration is valid; reference-bound training needs the corrected objective.
    legacy_setup = ModelSetup(source, decoded.config, 'nucleotide', 'alleles')
    with pytest.raises(ConfigurationError, match='Short-D'):
        prepare_training_data(legacy_setup)
    state = ResumeState({}, {}, {}, 2, {}, config=decoded.config, label_schema=corrected.label_schema())
    for candidate in (state, replace(state, config=None)):
        with pytest.raises(ConfigurationError, match='Short-D loss contract') as caught:
            validate_continuation(candidate, corrected.config, corrected.label_schema())
        assert caught.value.context['field'] == 'short_d_penalty'
