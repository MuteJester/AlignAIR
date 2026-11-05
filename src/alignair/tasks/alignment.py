"""The training-task contract, and the objective of the configured architecture."""
from __future__ import annotations

from typing import Protocol

import torch

from ..core.components import architecture_components
from ..core.config import AlignAIRConfig
from ..errors import ConfigurationError
from .metrics import MetricReport

LossResult = tuple[torch.Tensor, dict[str, torch.Tensor]]


class TrainingTasks(Protocol):
    """A trainer consumes task behavior without knowing the network's outputs."""
    def loss(self, outputs: dict, targets: dict) -> LossResult: ...
    def metrics(self, outputs: dict, targets: dict) -> dict[str, float]: ...
    def metric_report(self, outputs: dict, targets: dict) -> MetricReport: ...
    def selection_score(self, metrics: dict[str, float]) -> float: ...


def build_tasks(config: AlignAIRConfig, *, genes=None) -> TrainingTasks:
    """The configured architecture's own objective (``ModelComponents.objective``), which supervises its whole gene
    set."""
    if genes is not None:
        raise ConfigurationError("an architecture's own objective supervises its whole gene set",
                                 context={"architecture": config.architecture})
    return architecture_components(config.architecture).objective(config)
