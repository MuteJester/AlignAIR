"""The shared public training request, independent of sequence alphabet and result presentation."""
from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any

from ..core.config import AlignAIRConfig
from ..core.runtime import check_device
from ..data.flanks import FlankPolicy
from ..errors import ConfigurationError
from .guards import validate_data_workers, validate_training_request
from .records.novel import NovelAlleles
from .records.streams import CurriculumStreamPolicy
from .schedules import CosineSchedule

if TYPE_CHECKING:
    from GenAIRR import DataConfig

_PRESETS = {
    'quick': {'steps': 300, 'batch_size': 16, 'val_every': 100},
    'desktop': {'steps': 50_000, 'batch_size': 64, 'val_every': 2000},
    'full': {'steps': 300_000, 'batch_size': 128, 'val_every': 5000},
}


@dataclass(frozen=True)
class TrainingConfig:
    """Reference sources plus training and model settings, every value validated here, at construction
    (ConfigurationError; TrainingConfigError for the run's numbers): ``run_training`` checks only what needs the
    resolved reference.

    Model options are copied on construction and checked with ``max_seq_length`` and ``has_d`` as the model
    configuration they build (``AlignAIRConfig``); the allele counts come from the bound reference.
    ``data_workers`` > 0 generates training data in that many worker processes (one per curriculum
    stream at most: 28 for the default mix, 7 components of 4 streams); the batches are those of
    ``data_workers=0``, so it is not part of model identity.
    ``init_seed`` None seeds the model's initial weights from ``seed``. An int
    draws them from its own seed instead, and leaves the process generators (torch's CPU one, Python's and
    NumPy's) where ``seed`` left them: replicate runs of one ``seed`` then differ in their initial weights (and an
    accelerator's random draws, such as its dropout) while training on the same batches.
    Workers are spawned, so a script that trains with them needs an ``if __name__ == "__main__":``
    guard. On an accelerator, batches are assembled ahead from the CPU random state, so a CPU random
    draw in the training step or an ``on_iteration`` callback raises TrainingError;
    use ``data_workers=0`` there.
    ``flanks`` (``data.flanks.FlankPolicy``) puts library flanks (a 5' leader, a masked primer, a 3' constant
    region) around the curriculum's reads, which GenAIRR simulates bare: by default ``FlankPolicy()``, the
    chip's recipe; None trains on the bare reads.
    ``novel_alleles`` (``train.records.novel.NovelAlleles``) draws part of the reads from synthetic novel V alleles and
    puts their rows in each batch's bank (``data.novel_bank``), so the model does not memorise the training alleles:
    by default ``NovelAlleles()``, the shipped models' recipe; None trains on the dataconfigs' alleles alone, as does a
    run whose reference they cannot be drawn for (a dataconfig with no eligible V, loci of several species, or a custom
    reference, which names no species), which says so. The model and the saved reference stay the dataconfigs' either
    way.
    ``inverted_d`` (0 to 1) inverts the D of that share of the curriculum's records; 0, the default, trains on forward
    Ds alone.
    ``short_boost``, ``flanks``, ``novel_alleles`` and ``inverted_d`` are the run's settings of its record policy
    (``record_policy``), the rest of which is the recipe's.
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
    init_seed: int | None = None
    max_seq_length: int | None = None
    has_d: bool | None = None
    model_options: Mapping = field(default_factory=dict)
    resume_path: str | os.PathLike[str] | None = None
    schedule: CosineSchedule | None = None
    init_from: str | os.PathLike[str] | None = None
    data_workers: int = 0
    flanks: FlankPolicy | None = field(default_factory=FlankPolicy)
    novel_alleles: NovelAlleles | None = field(default_factory=NovelAlleles)
    inverted_d: float = 0.

    def __post_init__(self):
        if self.init_from and self.resume_path:
            raise ConfigurationError('choose init_from or resume_path')
        check_device(self.device)
        for name in ('resume_path', 'init_from'):
            value = getattr(self, name)
            if value is not None and not isinstance(value, (str, os.PathLike)):
                raise ConfigurationError(f'{name} must be a model file path or None (got {type(value).__name__})',
                                         context={'field': name})
        validate_data_workers(self.data_workers)
        validate_training_request(steps=self.steps, batch_size=self.batch_size, lr=self.lr, grad_clip=self.grad_clip)
        if not _uint32(self.seed) or self.init_seed is not None and not _uint32(self.init_seed):
            raise ConfigurationError('seed must be a uint32 and init_seed a uint32 or None',
                                     context={'fields': ['seed', 'init_seed']})
        if isinstance(self.val_every, bool) or not isinstance(self.val_every, int) or self.val_every < 0:
            raise ConfigurationError('val_every must be a nonnegative integer', context={'field': 'val_every'})
        policy = self.record_policy()                   # the policy validates its fields (short_boost .. inverted_d)
        object.__setattr__(self, 'short_boost', policy.short_boost)
        object.__setattr__(self, 'inverted_d', policy.inverted_d)
        if self.schedule is not None:
            if not isinstance(self.schedule, CosineSchedule):
                raise ConfigurationError('schedule must be a CosineSchedule or None')
            self.schedule.validate(self.lr)
        if isinstance(self.dataconfigs, (str, bytes)):
            raise ConfigurationError('dataconfigs must be a nonempty collection of reference sources')
        try:
            sources = tuple(self.dataconfigs)
        except TypeError as exc:
            raise ConfigurationError('dataconfigs must be an iterable of reference sources') from exc
        if not sources:
            raise ConfigurationError('dataconfigs must be nonempty')
        object.__setattr__(self, 'dataconfigs', sources)
        self._check_model_settings()

    def _check_model_settings(self) -> None:
        """``model_options`` (copied), ``max_seq_length`` and ``has_d``, checked as the model configuration they build
        (``AlignAIRConfig``); the window, the D gene and the locus count are not model options."""
        if not isinstance(self.model_options, Mapping):
            raise ConfigurationError('model_options must be a mapping of model configuration fields')
        object.__setattr__(self, 'model_options', deepcopy(dict(self.model_options)))
        protected = {'max_seq_length', 'has_d', 'num_chain_types'}
        if protected.intersection(self.model_options):
            raise ConfigurationError('model_options cannot set the window, the D gene or the locus count',
                                     context={'fields': sorted(protected.intersection(self.model_options))})
        unknown = sorted(set(self.model_options) - {option.name for option in fields(AlignAIRConfig)})
        if unknown:
            raise ConfigurationError('invalid model configuration options', context={'fields': unknown})
        window = {} if self.max_seq_length is None else {'max_seq_length': self.max_seq_length}
        d_gene = {} if self.has_d is None else {'has_d': self.has_d}
        AlignAIRConfig(**self.model_options, **window, **d_gene)    # the model settings' own checks, before any build

    def record_policy(self) -> CurriculumStreamPolicy:
        """The curriculum record policy the run draws its records with: the recipe's, with this configuration's
        ``short_boost``, ``flanks``, ``novel_alleles`` and ``inverted_d``."""
        return CurriculumStreamPolicy(short_boost=self.short_boost, flanks=self.flanks,
                                      novel_alleles=self.novel_alleles, inverted_d=self.inverted_d)

    @classmethod
    def from_genairr(cls, *dataconfigs: str | DataConfig, preset: str | None = None,
                     **overrides: Any) -> TrainingConfig:
        """A configuration training on GenAIRR ``dataconfigs``, from a named ``preset`` and ``overrides``."""
        if preset is not None and preset not in _PRESETS:
            raise ConfigurationError(f'unknown preset {preset!r}; choose from {sorted(_PRESETS)}')
        return cls(dataconfigs=tuple(dataconfigs), **{**_PRESETS.get(preset, {}), **overrides})

    def with_(self, **changes: Any) -> TrainingConfig:
        """A copy with ``changes``."""
        return replace(self, **changes)


def _uint32(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and 0 <= value < 2**32
