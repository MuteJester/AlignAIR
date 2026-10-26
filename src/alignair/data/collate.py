"""Collate existing (inputs, targets) samples through the shared batch contract."""
from .batching import TrainingSample, collate_samples


def align_collate(batch):
    samples = [TrainingSample(inputs["tokenized_sequence"], targets) for inputs, targets in batch]
    return tuple(collate_samples(samples))
