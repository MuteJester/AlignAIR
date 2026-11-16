"""Shared training mechanics: the engine, the loop, the run configuration and its workflow."""
from .config import TrainingConfig
from .engine import StepResult, TrainingEngine, ValidationResult
from .loop import TrainingIteration, training_iterations
from .records.novel import NovelAlleles
from .schedules import CosineSchedule
from .setup import OptimizerFactory, build_training_engine
from .workflow import TrainingRun, run_training

__all__ = [
    "CosineSchedule",
    "NovelAlleles",
    "OptimizerFactory",
    "StepResult",
    "TrainingConfig",
    "TrainingEngine",
    "TrainingIteration",
    "TrainingRun",
    "ValidationResult",
    "build_training_engine",
    "run_training",
    "training_iterations",
]
