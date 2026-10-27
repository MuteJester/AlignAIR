"""Shared training mechanics; experiment runners select data and study schedules."""
from .engine import TrainingEngine, StepResult, ValidationResult, evaluation_mode

__all__ = ["TrainingEngine", "StepResult", "ValidationResult", "evaluation_mode"]
