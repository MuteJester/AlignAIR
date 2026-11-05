"""Select record preparation and batch adapters from one reference-bound model setup."""
from __future__ import annotations

from dataclasses import dataclass

from ..core.setup import ModelSetup
from ..core.config import _reference_short_d_penalty
from ..errors import ConfigurationError
from .adapters import NucleotideBatchAdapter


@dataclass(frozen=True)
class TrainingDataSetup:
    batch_adapter: object
    validation_adapter: object

    def prepare_records(self, records, *, augment=False, seed=0, start_step=0):
        """The records as the batch adapters read them, in source order."""
        yield from records


def prepare_training_data(setup: ModelSetup, *, device='cpu') -> TrainingDataSetup:
    setup.validate()
    config = setup.config
    if config.short_d_penalty and not _reference_short_d_penalty(setup.reference,
            has_d=config.has_d, classification_mode=config.classification_mode):
        raise ConfigurationError('Short-D penalty requires the final D label to be Short-D',
                                 context={'field': 'short_d_penalty'},
                                 hint='Prepare a reference-bound configuration with short_d_penalty=False; '
                                      'use a weights-only warm start when changing a historical loss contract.')
    if config.classification_mode != 'multilabel':
        raise ConfigurationError('allele-record training requires multilabel heads')
    return TrainingDataSetup(NucleotideBatchAdapter(setup.reference, config, device),
                             NucleotideBatchAdapter(setup.reference, config, device, augment_orientation=False))
