"""Shared training mechanics; experiment runners select data and study schedules."""
from .engine import TrainingEngine, StepResult, ValidationResult, evaluation_mode
from .loop import TrainingIteration, training_iterations
from .setup import OptimizerFactory, build_training_engine

__all__ = ["TrainingEngine", "StepResult", "ValidationResult", "evaluation_mode",
           "TrainingIteration", "training_iterations", "OptimizerFactory", "build_training_engine"]
