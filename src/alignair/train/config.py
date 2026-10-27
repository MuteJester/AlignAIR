"""The shared public training request, independent of sequence alphabet and result presentation."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace

from ..errors import ConfigurationError

_PRESETS = {
    'quick': {'steps': 300, 'batch_size': 16, 'val_every': 100},
    'desktop': {'steps': 50_000, 'batch_size': 64, 'val_every': 2000},
    'full': {'steps': 300_000, 'batch_size': 128, 'val_every': 5000},
}


@dataclass(frozen=True)
class TrainingConfig:
    """Reference sources plus input/label choices and training/model settings.

    Model options are copied on construction and validated by reference-bound model setup.
    Label counts and input vocabulary are derived from the reference and selected spaces.
    """
    dataconfigs: tuple
    steps: int = 100_000
    batch_size: int = 64
    lr: float = 3e-4
    device: str = 'auto'
    short_boost: int = 1
    grad_clip: float | None = None
    val_every: int = 0
    seed: int = 0
    input_space: str = 'nucleotide'
    label_space: str = 'alleles'
    max_seq_length: int | None = None
    has_d: bool | None = None
    model_options: Mapping = field(default_factory=dict)
    resume_path: str | None = None

    def __post_init__(self):
        if self.input_space not in ('nucleotide', 'protein') or self.label_space not in ('alleles', 'protein_classes'):
            raise ConfigurationError('choose nucleotide/protein input and alleles/protein_classes labels')
        if self.label_space == 'protein_classes' and self.input_space != 'protein':
            raise ConfigurationError('the public protein-class trainer requires protein inputs')
        if isinstance(self.dataconfigs, (str, bytes)):
            raise ConfigurationError('dataconfigs must be a nonempty collection of reference sources')
        try:
            sources = tuple(self.dataconfigs)
        except TypeError as exc:
            raise ConfigurationError('dataconfigs must be an iterable of reference sources') from exc
        if not sources:
            raise ConfigurationError('dataconfigs must be nonempty')
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in (self.steps, self.batch_size, self.short_boost)):
            raise ConfigurationError('steps, batch_size and short_boost must be positive integers')
        if not isinstance(self.model_options, Mapping):
            raise ConfigurationError('model_options must be a mapping of model configuration fields')
        object.__setattr__(self, 'dataconfigs', sources)
        object.__setattr__(self, 'model_options', deepcopy(dict(self.model_options)))
        protected = {'input_space', 'label_space', 'max_seq_length', 'has_d', 'v_allele_count',
                     'j_allele_count', 'd_allele_count', 'v_label_count', 'j_label_count', 'd_label_count', 'num_chain_types'}
        if protected.intersection(self.model_options):
            raise ConfigurationError('model_options cannot override reference/input/label choices',
                                     context={'fields': sorted(protected.intersection(self.model_options))})

    @classmethod
    def from_genairr(cls, *dataconfigs, preset=None, **overrides):
        if preset is not None and preset not in _PRESETS:
            raise ConfigurationError(f'unknown preset {preset!r}; choose from {sorted(_PRESETS)}')
        return cls(dataconfigs=tuple(dataconfigs), **{**_PRESETS.get(preset, {}), **overrides})

    def with_(self, **changes):
        return replace(self, **changes)
