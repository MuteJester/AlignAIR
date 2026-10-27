"""Select record preparation and batch adapters from one reference-bound model setup."""
from __future__ import annotations

from dataclasses import dataclass

from ..core.setup import ModelSetup
from ..errors import ConfigurationError, TrainingDataError
from .adapters import CategoricalBatchAdapter, NucleotideBatchAdapter, ProteinAlleleBatchAdapter
from .sampling.protein import ProteinSampleBuilder, ProteinSamplePolicy
from .tokenizer import AAProteinTokenizer
from .augmentation.protein import ProteinSubstitutionPolicy


@dataclass(frozen=True)
class TrainingDataSetup:
    batch_adapter: object
    validation_adapter: object
    translated_only: bool
    sample_builder: ProteinSampleBuilder | None = None
    augmentation: ProteinSubstitutionPolicy | None = None

    def prepare_records(self, records, *, augment=False, seed=0):
        """Keep source order; fail on prolonged rejection rather than stall an infinite simulator."""
        import numpy as np
        rng = np.random.default_rng(seed) if augment and self.augmentation is not None else None
        rejected = 0
        for record in records:
            if self.sample_builder is None:
                yield record
            else:
                result = self.sample_builder(record)
                if result.sample is not None:
                    rejected = 0
                    sample = result.sample
                    if rng is not None:
                        sequence = self.augmentation.apply(record['sequence_aa'], rng)
                        tokens = self.sample_builder.tokenizer.encode(sequence)
                        if len(tokens) != len(sample['tokens']):
                            raise TrainingDataError('substitution augmentation must preserve residue positions')
                        sample = {**sample, 'tokens': tokens}
                    yield sample
                else:
                    rejected += 1
                    if rejected >= 1000:
                        raise TrainingDataError('no usable training sample after 1000 consecutive records',
                                                context={'stage': 'training_sample_preparation', 'reason': result.reason})


def prepare_training_data(setup: ModelSetup, *, device='cpu', augmentation=None) -> TrainingDataSetup:
    setup.validate()
    config = setup.config
    if augmentation is not None:
        if not isinstance(augmentation, ProteinSubstitutionPolicy):
            raise ConfigurationError('augmentation must be a ProteinSubstitutionPolicy or None')
        if (setup.input_space, setup.label_space) != ('protein', 'protein_classes'):
            raise ConfigurationError('substitution augmentation currently requires protein-class training')
    if setup.input_space == 'protein' and config.state_head:
        raise ConfigurationError('protein training presets do not provide nucleotide edit-state targets')
    if setup.label_space == 'protein_classes':
        if setup.input_space != 'protein' or config.classification_mode != 'categorical':
            raise ConfigurationError('protein-class training requires protein inputs and categorical heads')
        if config.use_meta_heads:
            raise ConfigurationError('protein-class training presets do not provide auxiliary-head targets')
        builder = ProteinSampleBuilder(setup.protein_reference.class_maps, AAProteinTokenizer(),
                                       config.max_seq_length,
                                       setup.reference.gene('D').index if config.has_d else None,
                                       ProteinSamplePolicy(require_productive=False, allow_stops=True))
        adapter = CategoricalBatchAdapter.from_config(config, device=device)
        return TrainingDataSetup(adapter, adapter, False, builder, augmentation)
    if config.classification_mode != 'multilabel':
        raise ConfigurationError('allele-record training requires multilabel heads')
    if setup.input_space == 'protein':
        adapter = ProteinAlleleBatchAdapter(setup.reference, config, device)
        return TrainingDataSetup(adapter, adapter, True)
    return TrainingDataSetup(NucleotideBatchAdapter(setup.reference, config, device),
                             NucleotideBatchAdapter(setup.reference, config, device, augment_orientation=False), False)
