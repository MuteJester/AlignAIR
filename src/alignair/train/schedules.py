"""Serializable learning-rate schedules."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import TYPE_CHECKING

from ..errors import ConfigurationError

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class CosineSchedule:
    """Cosine annealing from the initial learning rate to ``min_lr`` over ``total_steps`` updates."""
    total_steps: int
    min_lr: float = 1e-5

    def __post_init__(self):
        if isinstance(self.total_steps, bool) or not isinstance(self.total_steps, int) or self.total_steps <= 0:
            raise ConfigurationError('cosine total_steps must be a positive integer')
        if (isinstance(self.min_lr, bool) or not isinstance(self.min_lr, Real)
                or not math.isfinite(self.min_lr) or self.min_lr < 0):
            raise ConfigurationError('cosine min_lr must be a nonnegative finite number')
        object.__setattr__(self, 'min_lr', float(self.min_lr))

    def validate(self, lr: float) -> None:
        """Refuse an initial learning rate below ``min_lr`` or not finite."""
        if not isinstance(lr, Real) or isinstance(lr, bool) or not math.isfinite(lr) or lr < self.min_lr:
            raise ConfigurationError('initial learning rate must be finite and at least cosine min_lr')

    def build(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        """The torch scheduler of ``optimizer``."""
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=self.total_steps, eta_min=self.min_lr)

    def to_dict(self) -> dict:
        """The schedule as a plain dict, with its policy's name (a checkpoint records it)."""
        return {'policy': 'cosine-v1', **asdict(self)}
