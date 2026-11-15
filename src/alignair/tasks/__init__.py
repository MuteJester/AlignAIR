"""The training-task contract and its objective (``objective``), and the alignment metrics with explicit support
counts (``metrics``)."""
from .metrics import MetricReport
from .objective import TrainingTasks, build_tasks

__all__ = ["TrainingTasks", "build_tasks", "MetricReport"]
