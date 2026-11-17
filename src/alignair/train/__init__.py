"""Training: the run configuration (``TrainingConfig``, with its ``CosineSchedule`` and ``NovelAlleles`` settings),
``run_training`` and its ``TrainingRun``, and what a run's ``on_iteration`` callback receives (``TrainingIteration``,
whose ``result`` is a ``StepResult``). The engine and the loop are internal (``train.engine``, ``train.loop``)."""
from .config import TrainingConfig
from .engine import StepResult
from .loop import TrainingIteration
from .records.novel import NovelAlleles
from .schedules import CosineSchedule
from .workflow import TrainingRun, run_training

__all__ = [
    "CosineSchedule",
    "NovelAlleles",
    "StepResult",
    "TrainingConfig",
    "TrainingIteration",
    "TrainingRun",
    "run_training",
]
