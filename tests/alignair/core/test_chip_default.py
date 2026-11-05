"""The chip is every entry point's default: a reference alone prepares chip-v1 with the chip's settings, a light chain
prepares a chip without D units, and training defaults to the chip with library flanks."""
import itertools

import pytest

from alignair.api import predict_sequences
from alignair.core import prepare_model
from alignair.data.flanks import FlankPolicy
from alignair.reference.reference_set import ReferenceSet
from alignair.train.config import TrainingConfig

gd = pytest.importorskip("GenAIRR.data")

pytestmark = pytest.mark.integration


def test_a_reference_alone_prepares_the_chip():
    config = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)).config
    assert config.architecture == "chip-v1" and config.trunk_channels == 64
    assert config.boundary_mode == "half_open" and config.anchor_heads and config.d_presence_head
    assert config.chip_inputs and config.bank_orientation and not config.use_meta_heads


def test_a_light_chain_reference_prepares_a_chip_without_d_and_predicts():
    from alignair.train.gym.streams import CurriculumStreamPolicy, stream_records
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGK_OGRDB)
    setup = prepare_model(reference, trunk_channels=8)
    assert setup.config.architecture == "chip-v1"
    assert not setup.config.has_d and not setup.config.d_presence_head
    model = setup.build_model().eval()
    model.bind_reference(reference)
    component = CurriculumStreamPolicy().parameters_for(gd.HUMAN_IGK_OGRDB)[1]
    reads = [r["sequence"] for r in itertools.islice(stream_records(gd.HUMAN_IGK_OGRDB, component, 5), 3)]
    rows = predict_sequences(model, reference, reads, device="cpu", batch_size=4)
    assert len(rows) == 3 and all(not row.get("d_call") for row in rows)


def test_training_defaults_to_the_chip_with_library_flanks():
    config = TrainingConfig(dataconfigs=("HUMAN_IGH_OGRDB",))
    assert config.flanks == FlankPolicy() and "architecture" not in config.model_options
