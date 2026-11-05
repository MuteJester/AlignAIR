"""The training-task contract and the alignment metrics with explicit support counts."""
from .alignment import TrainingTasks, build_tasks
from .metrics import MetricAccumulator, MetricReport

__all__ = ["TrainingTasks", "build_tasks", "MetricAccumulator", "MetricReport"]
