"""The shared public training request, independent of sequence alphabet and result presentation."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace

from ..errors import ConfigurationError
from ..data.augmentation.protein import ProteinSubstitutionPolicy
from ..data.augmentation.partial import ProteinCropPolicy
from .schedules import CosineSchedule
from .study import TrainingStudy
from .guards import validate_data_workers

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
    ``data_workers`` > 0 generates training data in that many worker processes (one per curriculum
    component at most); the batches are those of ``data_workers=0``, so it is not part of model identity.
    Workers are spawned, so a script that trains with them needs an ``if __name__ == "__main__":``
    guard. On an accelerator, batches are assembled ahead from the CPU random state, so a CPU random
    draw in the training step, an ``on_iteration`` callback or a study evaluator raises TrainingError;
    use ``data_workers=0`` there.
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
    augmentation: ProteinSubstitutionPolicy | ProteinCropPolicy | None = None
    schedule: CosineSchedule | None = None
    study: TrainingStudy | None = None
    init_from: str | None = None
    trust_pickle: bool = False
    data_workers: int = 0

    def __post_init__(self):
        if self.init_from and self.resume_path or type(self.trust_pickle) is not bool:
            raise ConfigurationError('choose init_from or resume_path; trust_pickle must be a boolean')
        validate_data_workers(self.data_workers)
        if self.augmentation is not None and not isinstance(self.augmentation, (ProteinSubstitutionPolicy, ProteinCropPolicy)):
            raise ConfigurationError('augmentation must be a ProteinSubstitutionPolicy, ProteinCropPolicy or None')
        if self.schedule is not None:
            if not isinstance(self.schedule, CosineSchedule):
                raise ConfigurationError('schedule must be a CosineSchedule or None')
            self.schedule.validate(self.lr)
        if self.study is not None and not isinstance(self.study, TrainingStudy):
            raise ConfigurationError('study must be a TrainingStudy or None')
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
