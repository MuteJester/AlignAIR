"""A model's size by component (``sizes``): its parameters and Conv1d layers, read from its inspection points and its
core's units, grouped into trunk, parser, reader, scans, heads and other, and summed into one compact headline line."""
import pytest
import torch

import GenAIRR.data as gd

from alignair.core import prepare_model
from alignair.reference.reference_set import ReferenceSet
from alignair_lab.sizes import GROUPS, checkpoint_size, model_size, size_line

pytestmark = pytest.mark.component

UNITS = ('core.trunk', 'core.parser', 'core.anchors', 'core.reader', 'core.scans', 'core.d_presence')


@pytest.fixture(scope='module')
def reference():
    return ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)


def build(reference, **options):
    torch.manual_seed(0)
    return prepare_model(reference, **options).build_model()


def test_the_chips_size_by_component(reference):
    size = model_size(build(reference))
    assert (size['parameters'], size['conv1d']) == (314_812, 22)
    components = size['components']
    assert set(components) == {'embedding', 'orientation_head', *UNITS}     # the bound bank holds no weights
    assert components['core.trunk'] == {'parameters': 250_176, 'conv1d': 13}
    assert components['embedding'] == {'parameters': 18_624, 'conv1d': 0}
    groups = size['groups']
    assert list(groups) == list(GROUPS) and sum(groups.values()) == size['parameters']
    assert groups['trunk'] == 250_176 and groups['heads'] == (components['core.anchors']['parameters']
                                                             + components['core.d_presence']['parameters'])
    assert sum(component['conv1d'] for component in components.values()) == 22


def test_a_light_chain_chip_has_no_d_presence_unit(reference):
    size = model_size(build(ReferenceSet.from_dataconfigs(gd.HUMAN_IGK_OGRDB)))
    assert 'core.d_presence' not in size['components']
    assert sum(component['parameters'] for component in size['components'].values()) == size['parameters']


def test_the_layers_no_component_holds_are_unattributed(reference):
    """A Conv1d attached outside every component is counted as unattributed, its parameters and its layer, so the
    components still add up to the model's totals."""
    model = build(reference)
    assert 'unattributed' not in model_size(model)['components']
    model.stray = torch.nn.Conv1d(4, 2, 3)
    size = model_size(model)
    assert size['components']['unattributed'] == {'parameters': 4 * 2 * 3 + 2, 'conv1d': 1}
    assert sum(component['conv1d'] for component in size['components'].values()) == size['conv1d'] == 23
    assert sum(component['parameters'] for component in size['components'].values()) == size['parameters']


def test_a_model_file_is_sized_as_its_model(reference, tmp_path):
    from alignair.train.checkpoints import save_checkpoint
    model = build(reference, embed_dim=4, trunk_channels=8)
    path = tmp_path / 'model.alignair'
    save_checkpoint(path, model, reference, 1, dataconfigs=[gd.HUMAN_IGH_OGRDB])
    assert checkpoint_size(path) == {**model_size(model), 'architecture': 'chip-v1'}


def test_the_size_line_gives_the_groups_in_millions_the_conv_count_and_the_rate(reference):
    chip = {**model_size(build(reference)), 'architecture': 'chip-v1'}
    groups = chip['groups']
    assert size_line('baseline', chip, 5.31) == (
        'Size of the baseline (chip-v1): 0.315M parameters (trunk 0.250M, parser '
        f"{groups['parser'] / 1e6:.3f}M, reader {groups['reader'] / 1e6:.3f}M, scans {groups['scans'] / 1e6:.3f}M, "
        f"heads {groups['heads'] / 1e6:.3f}M, other {groups['other'] / 1e6:.3f}M), 22 Conv1d, 5.31 steps/s.")
    assert size_line('candidate', chip, None).endswith(', 22 Conv1d, steps/s not recorded.')
