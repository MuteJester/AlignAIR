"""The shared public training request, independent of sequence alphabet and result presentation."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace

from ..errors import ConfigurationError
from ..data.flanks import FlankPolicy
from .gym.novel import NovelAlleles
from .gym.streams import inverted_d_share
from .schedules import CosineSchedule
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
    stream at most: 28 for the default mix, 7 components of 4 streams); the batches are those of
    ``data_workers=0``, so it is not part of model identity.
    ``init_seed`` None seeds the model's initial weights from ``seed``, as every run before it did. An int
    draws them from its own seed instead, and leaves the process generators (torch's CPU one, Python's and
    NumPy's) where ``seed`` left them: replicate runs of one ``seed`` then differ in their initial weights (and an
    accelerator's random draws, such as its dropout) while training on the same batches.
    Workers are spawned, so a script that trains with them needs an ``if __name__ == "__main__":``
    guard. On an accelerator, batches are assembled ahead from the CPU random state, so a CPU random
    draw in the training step, an ``on_iteration`` callback or a study evaluator raises TrainingError;
    use ``data_workers=0`` there.
    ``flanks`` (``data.flanks.FlankPolicy``) puts library flanks (a 5' leader, a masked primer, a 3' constant
    region) around the curriculum's reads, which GenAIRR simulates bare: by default ``FlankPolicy()``, the
    chip's recipe; None trains on the bare reads.
    ``novel_alleles`` (``train.gym.novel.NovelAlleles``) draws part of the reads from synthetic novel V alleles and
    puts their rows in each batch's bank (``data.novel_bank``); None, the default, trains on the dataconfigs' alleles
    alone. The model and the saved reference stay the dataconfigs' either way.
    ``inverted_d`` (0 to 1) inverts the D of that share of the curriculum's records
    (``gym.streams.CurriculumStreamPolicy``); 0, the default, trains on forward Ds alone.
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
    resume_path: str | None = None
    schedule: CosineSchedule | None = None
    init_from: str | None = None
    data_workers: int = 0
    flanks: FlankPolicy | None = field(default_factory=FlankPolicy)
    novel_alleles: NovelAlleles | None = None
    inverted_d: float = 0.

    def __post_init__(self):
        if self.init_from and self.resume_path:
            raise ConfigurationError('choose init_from or resume_path')
        validate_data_workers(self.data_workers)
        if self.flanks is not None and not isinstance(self.flanks, FlankPolicy):
            raise ConfigurationError('flanks must be a FlankPolicy or None', context={'field': 'flanks'})
        if self.novel_alleles is not None and not isinstance(self.novel_alleles, NovelAlleles):
            raise ConfigurationError('novel_alleles must be a NovelAlleles or None', context={'field': 'novel_alleles'})
        object.__setattr__(self, 'inverted_d', inverted_d_share(self.inverted_d))
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
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in (self.steps, self.batch_size, self.short_boost)):
            raise ConfigurationError('steps, batch_size and short_boost must be positive integers')
        if not isinstance(self.model_options, Mapping):
            raise ConfigurationError('model_options must be a mapping of model configuration fields')
        object.__setattr__(self, 'dataconfigs', sources)
        object.__setattr__(self, 'model_options', deepcopy(dict(self.model_options)))
        protected = {'max_seq_length', 'has_d', 'num_chain_types'}
        if protected.intersection(self.model_options):
            raise ConfigurationError('model_options cannot set the window, the D gene or the locus count',
                                     context={'fields': sorted(protected.intersection(self.model_options))})

    @classmethod
    def from_genairr(cls, *dataconfigs, preset=None, **overrides):
        if preset is not None and preset not in _PRESETS:
            raise ConfigurationError(f'unknown preset {preset!r}; choose from {sorted(_PRESETS)}')
        return cls(dataconfigs=tuple(dataconfigs), **{**_PRESETS.get(preset, {}), **overrides})

    def with_(self, **changes):
        return replace(self, **changes)
