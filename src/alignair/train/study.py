"""Study observation and retained checkpoint schedules, independent of the model's alphabet."""
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ..core.runtime import evaluation_mode
from ..errors import ConfigurationError


@dataclass(frozen=True)
class TrainingStudy:
    checkpoint_dir: str | None = None
    checkpoint_steps: tuple[int, ...] = ()
    validation_every: int = 0
    evaluate: Callable | None = None
    plateau_window: int = 0

    def __post_init__(self):
        try:
            steps = tuple(self.checkpoint_steps)
        except TypeError as exc:
            raise ConfigurationError('study checkpoint_steps must be an iterable of positive integers') from exc
        if (any(isinstance(step, bool) or not isinstance(step, int) or step <= 0 for step in steps)
                or len(set(steps)) != len(steps) or steps and not self.checkpoint_dir):
            raise ConfigurationError('study checkpoints require unique positive steps and a checkpoint directory')
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (self.validation_every, self.plateau_window)):
            raise ConfigurationError('study validation interval and plateau window must be nonnegative integers')
        if (self.evaluate is not None and not callable(self.evaluate)
                or self.validation_every and self.evaluate is None):
            raise ConfigurationError('study validation requires an evaluation callback')
        object.__setattr__(self, 'checkpoint_steps', steps)

    def to_dict(self):
        return {'checkpoint_steps': list(self.checkpoint_steps), 'validation_every': self.validation_every,
                'plateau_window': self.plateau_window}


class StudySession:
    def __init__(self, plan, model, save):
        self.plan, self.model, self.save = plan, model, save
        self.checkpoints = {}
        self.validations = []

    def observe(self, iteration):
        step, plan = iteration.step, self.plan
        if step in plan.checkpoint_steps:
            path = Path(plan.checkpoint_dir) / f'checkpoint-{step}.alignair'
            self.save(path, step)
            self.checkpoints[step] = str(path)
        if plan.validation_every and step % plan.validation_every == 0:
            with evaluation_mode(self.model):
                result = plan.evaluate(self.model, step)
            if not isinstance(result, Mapping):
                raise ConfigurationError('study evaluation must return a metric mapping')
            self.validations.append(dict(result))
            values = [row.get('objective') for row in self.validations[-plan.plateau_window:]] if plan.plateau_window else []
            if (values and len(values) == plan.plateau_window and all(value is not None for value in values)
                    and all(values[i] <= values[i - 1] for i in range(1, len(values)))):
                print(f'[plateau] validation objective has not improved over the last {plan.plateau_window} '
                      f'evaluations (step {step}).', flush=True)
