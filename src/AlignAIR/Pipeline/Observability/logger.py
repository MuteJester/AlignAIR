"""PipelineLogger — structured logging with automatic stage timing."""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StageMetrics:
    """Metrics collected during one stage execution."""

    name: str = ""
    index: int = 0
    wall_seconds: float = 0.0
    skipped: bool = False
    custom: Dict[str, Any] = field(default_factory=dict)


class PipelineLogger:
    """Structured logger with automatic per-stage timing.

    Emits human-readable lines via stdlib logging AND collects
    machine-readable StageMetrics for the run report.
    """

    def __init__(self, name: str = "AlignAIR.Pipeline"):
        self._log = logging.getLogger(name)
        self._stage_metrics: List[StageMetrics] = []
        self._run_start: float = 0.0
        self._run_id: str = ""
        self._config_summary: Dict[str, Any] = {}

    def start_run(self, run_id: str, config_summary: Optional[Dict[str, Any]] = None) -> None:
        self._run_id = run_id
        self._config_summary = config_summary or {}
        self._run_start = time.monotonic()
        self._stage_metrics.clear()
        self._log.info("Pipeline run %s starting", run_id)

    @contextmanager
    def stage(self, name: str, index: int, total: int):
        """Context manager that times a stage and captures metrics.

        Usage::

            with logger.stage("BatchInference", 2, 9) as metrics:
                # ... do work ...
                metrics.custom["batches_processed"] = 42
        """
        metrics = StageMetrics(name=name, index=index)
        self._log.info(
            "Stage %d/%d '%s' STARTING...",
            index + 1, total, name,
        )
        t0 = time.monotonic()
        try:
            yield metrics
        finally:
            metrics.wall_seconds = time.monotonic() - t0
            self._stage_metrics.append(metrics)
            if metrics.skipped:
                self._log.info(
                    "Stage %d/%d '%s' SKIPPED (condition not met)",
                    index + 1, total, name,
                )
            else:
                self._log.info(
                    "Stage %d/%d '%s' COMPLETED in %.2fs",
                    index + 1, total, name, metrics.wall_seconds,
                )

    def skip_stage(self, name: str, index: int, total: int) -> None:
        """Record a skipped stage without a context manager."""
        metrics = StageMetrics(name=name, index=index, skipped=True)
        self._stage_metrics.append(metrics)
        self._log.info(
            "Stage %d/%d '%s' SKIPPED (condition not met)",
            index + 1, total, name,
        )

    def log(self, message: str, *args, level: int = logging.INFO, **kwargs) -> None:
        self._log.log(level, message, *args, **kwargs)

    def finish_run(self) -> Dict[str, Any]:
        """Return the complete run report."""
        total_wall = time.monotonic() - self._run_start

        stage_breakdown = []
        for m in self._stage_metrics:
            entry = {
                "name": m.name,
                "index": m.index,
                "wall_seconds": round(m.wall_seconds, 4),
                "skipped": m.skipped,
            }
            if m.custom:
                entry["custom"] = m.custom
            stage_breakdown.append(entry)

        report = {
            "run_id": self._run_id,
            "total_wall_seconds": round(total_wall, 4),
            "stages_total": len(self._stage_metrics),
            "stages_executed": sum(1 for m in self._stage_metrics if not m.skipped),
            "stages_skipped": sum(1 for m in self._stage_metrics if m.skipped),
            "stage_breakdown": stage_breakdown,
            "config_summary": self._config_summary,
        }

        self._log.info("Pipeline completed in %.2fs", total_wall)
        return report

    @property
    def stage_timings(self) -> Dict[str, float]:
        """Map of stage name → wall seconds for provenance."""
        return {m.name: m.wall_seconds for m in self._stage_metrics}

    @property
    def stages_executed(self) -> List[str]:
        """Ordered list of stage names that actually ran."""
        return [m.name for m in self._stage_metrics if not m.skipped]
