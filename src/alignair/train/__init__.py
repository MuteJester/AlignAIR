"""Shared training mechanics; experiment runners select data and study schedules."""
from .engine import TrainingEngine, StepResult, ValidationResult, evaluation_mode
from .loop import TrainingIteration, training_iterations
from .setup import OptimizerFactory, build_training_engine
from .config import TrainingConfig
from .workflow import TrainingRun, run_training
from .schedules import CosineSchedule
from .study import TrainingStudy
from .gym.novel import NovelAlleles

__all__ = ["TrainingEngine", "StepResult", "ValidationResult", "evaluation_mode",
           "TrainingIteration", "training_iterations", "OptimizerFactory", "build_training_engine",
           "TrainingConfig", "TrainingRun", "run_training", "CosineSchedule", "TrainingStudy", "NovelAlleles"]
