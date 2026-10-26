"""Shared alignment losses, metrics, and training-task contracts."""
from .alignment import AlignmentTasks, TrainingTasks, build_tasks
from .metrics import MetricAccumulator, MetricReport

__all__ = ["AlignmentTasks", "TrainingTasks", "build_tasks", "MetricAccumulator", "MetricReport"]
