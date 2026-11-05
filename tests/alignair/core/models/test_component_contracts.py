"""Architecture extension boundaries: isolated configurations and registered component contracts."""
from dataclasses import replace

import pytest
import torch

from alignair.core import AlignAIR, ModelComponents, build_model_config, register_architecture
from alignair.core import components
from alignair.core.gene_branch import GeneBranch
from alignair.core.layers import TokenAndPositionEmbedding
from alignair.errors import ConfigurationError, current_error_context

pytestmark = pytest.mark.component


def config(**options):
    return build_model_config(3, 2, max_seq_length=128, embed_dim=4, filters=4, block_out=8, **options)


@pytest.fixture
def architecture(monkeypatch):
    monkeypatch.setattr(components, '_ARCHITECTURES', dict(components._ARCHITECTURES))
    def register(factories):
        name = 'test-component-contract-v1'
        register_architecture(name, factories)
        return name
    return register


@pytest.mark.parametrize('protein', [False])
def test_model_configuration_is_an_independent_snapshot_including_nested_values(architecture, protein):
    received = []
    def embedding(cfg):
        received.append(cfg)
        return TokenAndPositionEmbedding(cfg.vocab_size, cfg.embed_dim, cfg.max_seq_length)
    name = architecture(replace(ModelComponents(), embedding=embedding))
    cfg = config(input_space='protein' if protein else 'nucleotide',
                 label_space='protein_classes' if protein else 'alleles',
                 architecture=name, allele_temperatures={'v': 1.5})
    first, second = AlignAIR(cfg).eval(), AlignAIR(cfg).eval()
    assert received[0] is first.cfg and received[1] is second.cfg
    assert first.cfg == second.cfg == cfg
    assert first.cfg is not cfg and first.cfg is not second.cfg
    cfg.max_seq_length = 256
    cfg.classification_mode = 'multilabel' if protein else 'categorical'
    cfg.allele_temperatures['v'] = 9.
    assert first.cfg.max_seq_length == second.cfg.max_seq_length == 128
    assert first.cfg.classification_mode == ('categorical' if protein else 'multilabel')
    assert first.cfg.allele_temperatures == second.cfg.allele_temperatures == {'v': 1.5}
    first.cfg.allele_temperatures['v'] = 2.
    assert second.cfg.allele_temperatures['v'] == 1.5 and cfg.allele_temperatures['v'] == 9.
    assert all(left is not right for left, right in zip(first.parameters(), second.parameters()))
    tokens = torch.zeros(2, 128, dtype=torch.long)
    tokens[:, :64] = 1
    with torch.no_grad():
        output = first({'tokenized_sequence': tokens})
    assert output[first.cfg.classification_key('v')].shape == (2, 3)


@pytest.mark.parametrize('cfg', [None, {}, object()])
def test_bad_configuration_type_fails_before_factories(cfg):
    with pytest.raises(ConfigurationError) as caught:
        AlignAIR(cfg)
    assert caught.value.context['field'] == 'config'


@pytest.mark.parametrize('factory,point', [('embedding', 'embedding'), ('gene_branch', 'branches.v'),
    ('orientation_head', 'orientation_head'), ('meta_tower', 'meta_tower'), ('state_head', 'state_branch')])
def test_bad_factory_results_identify_the_component(architecture, factory, point):
    name = architecture(replace(ModelComponents(), **{factory: lambda *args: None}))
    with pytest.raises(ConfigurationError, match='torch module') as caught:
        AlignAIR(config(architecture=name, state_head=True))
    assert caught.value.context['component'] == point
    assert caught.value.context['stage'] == 'model_construction'
    assert caught.value.context['returned_type'] == 'NoneType'
    assert current_error_context() == {}


@pytest.mark.parametrize('method', ['segment', 'classify', 'component_modules'])
def test_incomplete_gene_branches_fail_during_assembly(architecture, method):
    def branch(spec, cfg):
        result = GeneBranch(spec, cfg)
        setattr(result, method, None)
        return result
    name = architecture(replace(ModelComponents(), gene_branch=branch))
    with pytest.raises(ConfigurationError) as caught:
        AlignAIR(config(architecture=name))
    assert caught.value.context['methods'] == [method]
    assert caught.value.context['component'] == 'branches.v'


@pytest.mark.parametrize('field', ['segmentation_tower', 'classification_tower'])
def test_replacement_towers_require_registered_torch_modules(field):
    cfg = config()
    with pytest.raises(ConfigurationError) as caught:
        GeneBranch(cfg.gene_specs[0], cfg, **{field: object()})
    assert caught.value.context['field'] == field


@pytest.mark.parametrize('heads', [None, [], {'bad': None}, {'': torch.nn.Identity()},
    {1: torch.nn.Identity()}, {'bad.name': torch.nn.Identity()}, {'training': torch.nn.Identity()}])
def test_malformed_head_mappings_are_contextual_configuration_errors(architecture, heads):
    name = architecture(replace(ModelComponents(), meta_heads=lambda cfg: heads))
    with pytest.raises(ConfigurationError) as caught:
        AlignAIR(config(architecture=name))
    assert caught.value.context['component'] == 'meta_heads'
    assert caught.value.context['stage'] == 'model_construction'


def test_disabled_capabilities_do_not_call_their_factories(architecture):
    def forbidden(cfg):
        pytest.fail('disabled component factory called')
    name = architecture(replace(ModelComponents(), orientation_head=forbidden, meta_tower=forbidden,
                                meta_heads=forbidden, state_head=forbidden))
    model = AlignAIR(config(use_orientation=False, bank_orientation=False, use_meta_heads=False, anchor_heads=False,
                            d_presence_head=False, architecture=name))
    assert model.orientation_head is model.meta_tower is model.state_branch is None
    assert not model.meta_heads


def test_registered_module_dict_is_a_valid_head_factory_result(architecture):
    heads = torch.nn.ModuleDict({'custom': torch.nn.Identity()})
    name = architecture(replace(ModelComponents(), meta_heads=lambda cfg: heads))
    model = AlignAIR(config(architecture=name))
    assert model.meta_heads['custom'] is heads['custom']
    model.clamp_params()  # Replacement heads need not implement the optional constraint hook.


@pytest.mark.parametrize('name', [None, '', ' leading', 'trailing '])
def test_architecture_names_are_portable_unambiguous_identifiers(name):
    with pytest.raises(ConfigurationError, match='architecture name'):
        register_architecture(name, ModelComponents())


def test_registration_rejects_untyped_factory_sets():
    with pytest.raises(ConfigurationError, match='ModelComponents'):
        register_architecture('not-registered', None)


@pytest.mark.parametrize('points', [None, [], {'': torch.nn.Identity()}, {'foreign': torch.nn.Identity()},
                                  {1: torch.nn.Identity()}, {'bad': object()}])
def test_invalid_inspection_points_cannot_escape_module_ownership(points):
    model = AlignAIR(config())
    model.branches['v'].component_modules = lambda: points
    with pytest.raises(ConfigurationError) as caught:
        model.component_modules()
    assert caught.value.context['component'] == 'branches.v'
    assert caught.value.context['stage'] == 'component_inspection'
    assert current_error_context() == {}


def test_inspection_names_are_aliases_for_actual_registered_modules():
    model = AlignAIR(config())
    model.branches['v'].component_modules = lambda: {'shared_embedding': model.embedding}
    assert model.component_modules()['v.shared_embedding'] is model.embedding
