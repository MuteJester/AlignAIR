"""Unknown productivity is masked out of the productive objective; the historical objective trained it as False."""
from dataclasses import asdict, replace
import json

import pytest
import torch

from alignair.core import AlignAIR, AlignAIRConfig, prepare_model
from alignair.errors import ConfigurationError
from alignair.model_file import container, load_model, read_model_state, serialize
from alignair.reference import GeneReference, ReferenceSet
from alignair.train.checkpoints import ResumeState, validate_continuation

pytestmark = pytest.mark.component


def reference():
    # The final Short-D label keeps the Short-D objective equal on both sides of every continuation below.
    panels = {'V': (['IGHV1*01'], ['ATGTGT']), 'J': (['IGHJ1*01'], ['TGGAAA']),
              'D': (['IGHD1*01', 'Short-D'], ['GGG', 'NNN'])}
    return ReferenceSet({gene: GeneReference(names, sequences, {name: i for i, name in enumerate(names)})
                         for gene, (names, sequences) in panels.items()}, has_d=True)


def test_reference_bound_models_mask_unknown_productivity_and_historical_configs_keep_the_old_mapping():
    source = reference()
    assert prepare_model(source).config.productive_unknown == 'masked'
    assert AlignAIRConfig.from_reference(source).productive_unknown == 'masked'
    assert AlignAIRConfig(2, 2).productive_unknown == 'negative'
    assert prepare_model(source, productive_unknown='negative').config.productive_unknown == 'negative'
    # Without metadata heads the objective is inactive: the historical value keeps old checkpoints resumable.
    assert prepare_model(source, use_meta_heads=False).config.productive_unknown == 'negative'
    assert AlignAIRConfig.from_reference(source, use_meta_heads=False).productive_unknown == 'negative'


@pytest.mark.parametrize('value', [None, True, 'unknown'])
def test_productive_label_policy_is_validated(value):
    with pytest.raises(ConfigurationError) as caught:
        AlignAIRConfig(2, 2, productive_unknown=value)
    assert caught.value.context['field'] == 'productive_unknown'


def test_historical_model_loads_for_inference_but_cannot_continue_into_the_masked_objective(tmp_path):
    source = reference()
    masked = prepare_model(source, max_seq_length=128, embed_dim=4, filters=4, block_out=8)
    old_config = replace(masked.config, productive_unknown='negative')
    old_model = AlignAIR(old_config).eval()
    encoded = asdict(old_config)
    encoded.pop('productive_unknown')  # A historical artifact has no explicit policy.
    path = tmp_path / 'historical.alignair'
    container.write_container(path, {'model': {'label_schema': masked.label_schema()}}, {
        'config': (json.dumps(encoded).encode(), 'zlib'),
        'weights': (serialize.state_dict_to_bytes(old_model.state_dict()), 'none'),
        'reference_json': (serialize.reference_to_json(source), 'zlib'),
    })
    decoded, restored = read_model_state(path), load_model(path).model
    assert decoded.config.productive_unknown == restored.cfg.productive_unknown == 'negative'
    tokens = torch.zeros(1, 128, dtype=torch.long)
    tokens[:, :8] = 1
    with torch.no_grad():
        before, after = old_model({'tokenized_sequence': tokens}), restored({'tokenized_sequence': tokens})
    assert before.keys() == after.keys()
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    state = ResumeState({}, {}, {}, 2, {}, config=decoded.config, label_schema=masked.label_schema())
    for candidate in (state, replace(state, config=None)):
        with pytest.raises(ConfigurationError, match='productive') as caught:
            validate_continuation(candidate, masked.config, masked.label_schema())
        assert caught.value.context['field'] == 'productive_unknown'
        assert 'init_from' in caught.value.hint
    validate_continuation(replace(state, config=masked.config), masked.config, masked.label_schema())
