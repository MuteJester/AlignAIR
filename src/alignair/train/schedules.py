"""Serializable learning-rate policies for the shared training lifecycle."""
from dataclasses import asdict, dataclass
import math
from numbers import Real

from ..errors import ConfigurationError


@dataclass(frozen=True)
class CosineSchedule:
    total_steps: int
    min_lr: float = 1e-5

    def __post_init__(self):
        if isinstance(self.total_steps, bool) or not isinstance(self.total_steps, int) or self.total_steps <= 0:
            raise ConfigurationError('cosine total_steps must be a positive integer')
        if (isinstance(self.min_lr, bool) or not isinstance(self.min_lr, Real)
                or not math.isfinite(self.min_lr) or self.min_lr < 0):
            raise ConfigurationError('cosine min_lr must be a nonnegative finite number')
        object.__setattr__(self, 'min_lr', float(self.min_lr))

    def validate(self, lr):
        if not isinstance(lr, Real) or isinstance(lr, bool) or not math.isfinite(lr) or lr < self.min_lr:
            raise ConfigurationError('initial learning rate must be finite and at least cosine min_lr')

    def build(self, optimizer):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=self.total_steps, eta_min=self.min_lr)

    def to_dict(self):
        return {'policy': 'cosine-v1', **asdict(self)}
