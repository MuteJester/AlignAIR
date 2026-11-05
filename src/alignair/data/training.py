"""The batch adapters of one reference-bound model setup."""
from __future__ import annotations

from dataclasses import dataclass

from ..core.setup import ModelSetup
from .adapters import NucleotideBatchAdapter


@dataclass(frozen=True)
class TrainingDataSetup:
    batch_adapter: object
    validation_adapter: object


def prepare_training_data(setup: ModelSetup, *, device='cpu') -> TrainingDataSetup:
    setup.validate()
    config = setup.config
    return TrainingDataSetup(NucleotideBatchAdapter(setup.reference, config, device),
                             NucleotideBatchAdapter(setup.reference, config, device, augment_orientation=False))
