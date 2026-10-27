"""Shared training mechanics; experiment runners select data and study schedules."""
from .engine import TrainingEngine, StepResult, ValidationResult, evaluation_mode
from .loop import TrainingIteration, training_iterations
from .setup import OptimizerFactory, build_training_engine
from .diagnostics import overfit_one_batch
from .config import TrainingConfig
from .workflow import TrainingRun, run_training
from .schedules import CosineSchedule
from .study import TrainingStudy

__all__ = ["TrainingEngine", "StepResult", "ValidationResult", "evaluation_mode",
           "TrainingIteration", "training_iterations", "OptimizerFactory", "build_training_engine",
           "overfit_one_batch", "TrainingConfig", "TrainingRun", "run_training", "CosineSchedule", "TrainingStudy"]
