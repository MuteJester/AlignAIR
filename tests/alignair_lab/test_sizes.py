"""A model's size by component (``sizes``): its parameters and Conv1d layers, read from its inspection points, grouped
into segmentation, trunk, classification, meta and other, and summed into one compact headline line."""
import pytest
import torch

import GenAIRR.data as gd

from alignair.core import prepare_model
from alignair.reference.reference_set import ReferenceSet
from alignair_lab.sizes import GROUPS, checkpoint_size, model_size, size_line

pytestmark = pytest.mark.component

GENES = ('v', 'd', 'j')


@pytest.fixture(scope='module')
def reference():
    return ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)


def build(reference, architecture, **options):
    torch.manual_seed(0)
    return prepare_model(reference, architecture=architecture, **options).build_model()


def test_conv_v1s_size_by_component(reference):
    size = model_size(build(reference, 'conv-v1'))
    assert (size['parameters'], size['conv1d']) == (14_306_909, 104)
    components = size['components']
    assert set(components) == {'embedding', 'orientation_head', 'meta_tower', 'meta_heads.mutation_rate',
                               'meta_heads.indel_count', 'meta_heads.productive',
                               *(f'{g}.{part}' for g in GENES for part in ('segmentation', 'classification'))}
    for g in GENES:   # a segmentation tower (13 Conv1d) and two Linear(576, 576) boundary heads
        assert components[f'{g}.segmentation'] == {'parameters': 1_854_656 + 2 * 332_352, 'conv1d': 13}
    assert components['meta_tower'] == {'parameters': 1_854_656, 'conv1d': 13}
    assert components['embedding'] == {'parameters': 18_624, 'conv1d': 0}
    groups = size['groups']
    assert list(groups) == list(GROUPS) and sum(groups.values()) == size['parameters']
    assert (groups['segmentation'], groups['trunk']) == (7_558_080, 0)
    assert groups['meta'] == 1_854_656 + sum(components[f'meta_heads.{name}']['parameters']
                                            for name in ('mutation_rate', 'indel_count', 'productive'))
    assert sum(component['conv1d'] for component in components.values()) == 104


def test_lab_trunk_v1s_size_by_component(reference):
    import alignair_lab.architectures  # noqa: F401  (registers lab-trunk-v1)
    size = model_size(build(reference, 'lab-trunk-v1'))
    assert (size['parameters'], size['conv1d']) == (7_741_475, 81)
    components = size['components']
    assert components['trunk'] == {'parameters': 991_872, 'conv1d': 13}
    for g in GENES:   # the 1x1 boundary convolution
        assert components[f'{g}.segmentation'] == {'parameters': 258, 'conv1d': 1}
    groups = size['groups']
    assert (groups['segmentation'], groups['trunk']) == (774, 991_872)
    assert sum(groups.values()) == size['parameters']


def test_a_model_file_is_sized_as_its_model(reference, tmp_path):
    from alignair.train.checkpoints import save_checkpoint
    model = build(reference, 'conv-v1', embed_dim=4, filters=4, block_out=8)
    path = tmp_path / 'model.alignair'
    save_checkpoint(path, model.cfg, model, None, 1, dataconfigs=[gd.HUMAN_IGH_OGRDB])
    assert checkpoint_size(path) == {**model_size(model), 'architecture': 'conv-v1'}


def test_the_size_line_gives_the_groups_in_millions_the_conv_count_and_the_rate(reference):
    import alignair_lab.architectures  # noqa: F401  (registers lab-trunk-v1)
    conv = {**model_size(build(reference, 'conv-v1')), 'architecture': 'conv-v1'}
    trunk = {**model_size(build(reference, 'lab-trunk-v1')), 'architecture': 'lab-trunk-v1'}
    assert size_line('baseline', conv, 5.31) == (
        'Size of the baseline (conv-v1): 14.307M parameters (segmentation 7.558M, classification '
        f"{conv['groups']['classification'] / 1e6:.3f}M, meta {conv['groups']['meta'] / 1e6:.3f}M, other "
        f"{conv['groups']['other'] / 1e6:.3f}M), 104 Conv1d, 5.31 steps/s.")
    line = size_line('candidate', trunk, None)
    assert line.startswith('Size of the candidate (lab-trunk-v1): 7.741M parameters (segmentation 0.001M, trunk '
                           '0.992M, classification ')
    assert line.endswith(', 81 Conv1d, steps/s not recorded.')
