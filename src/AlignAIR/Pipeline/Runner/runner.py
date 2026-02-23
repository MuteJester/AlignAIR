"""Pipeline runner — orchestrates stage execution with validation and observability."""
from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional

from AlignAIR.Pipeline.Stage.protocol import Stage, ConditionalStage
from AlignAIR.Pipeline.Runner.slot_store import SlotStore, OVERWRITABLE_SLOTS
from AlignAIR.Pipeline.Models.config import PipelineConfig
from AlignAIR.Pipeline.Errors.exceptions import ConfigError
from AlignAIR.Pipeline.Observability.logger import PipelineLogger


# Memory release schedule: after completing a stage, release these slots
RELEASE_SCHEDULE: Dict[str, List[str]] = {
    "CleanAndExtract": ["raw_predictions"],
}


class AlignAIRPipeline:
    """Orchestrates the execution of pipeline stages.

    Responsibilities:
    1. Pre-flight validation (DAG check — all reads satisfied by prior writes)
    2. Stage execution with typed context injection
    3. Output validation (all declared writes produced)
    4. Per-stage timing and structured logging
    5. Memory release scheduling
    6. Run report generation
    """

    def __init__(
        self,
        config: PipelineConfig,
        stages: List[Stage],
        logger: Optional[PipelineLogger] = None,
    ):
        self.config = config
        self.stages = stages
        self.logger = logger or PipelineLogger()

    def validate(self) -> None:
        """Pre-flight validation: ensure the stage DAG is satisfiable.

        Checks that every stage's `reads` is a subset of the cumulative
        `writes` of all preceding stages plus the initial 'config' slot.
        """
        available: set = {"config"}

        for i, stage in enumerate(self.stages):
            missing = stage.reads - available
            if missing:
                raise ConfigError(
                    f"Stage {i} '{stage.name}' requires slots {missing} "
                    f"which are not produced by any preceding stage. "
                    f"Available at this point: {sorted(available)}"
                )
            available |= stage.writes

    def run(self, sequences: Optional[List[str]] = None) -> SlotStore:
        """Execute all stages and return the final slot store.

        Args:
            sequences: Optionally pass sequences directly (for programmatic API).
                       If None, stages will load from config.sequences_path.

        Returns:
            SlotStore with all pipeline outputs.
        """
        self.validate()

        run_id = uuid.uuid4().hex[:12]
        self.logger.start_run(run_id, config_summary={
            "model_dir": self.config.model_dir,
            "output_format": self.config.output_format.value,
            "batch_size": self.config.memory.batch_size,
            "n_stages": len(self.stages),
        })

        store = SlotStore()
        store.set("config", self.config)
        if sequences is not None:
            store.set("sequences", sequences)

        total = len(self.stages)

        for i, stage in enumerate(self.stages):
            # Check conditional skip
            if isinstance(stage, ConditionalStage):
                context = store.build_context(stage.reads)
                if not stage.should_run(context):
                    self.logger.skip_stage(stage.name, i, total)
                    continue

            # Build typed context with only declared reads
            context = store.build_context(stage.reads)

            with self.logger.stage(stage.name, i, total) as metrics:
                stage.validate_inputs(context)
                outputs = stage.run(context)

                # Validate outputs
                if outputs:
                    stage.validate_outputs(outputs)

                # Store outputs
                for key, value in outputs.items():
                    allow_overwrite = key in OVERWRITABLE_SLOTS
                    store.set(key, value, allow_overwrite=allow_overwrite)

            # Memory release
            for key in RELEASE_SCHEDULE.get(stage.name, []):
                if store.has(key):
                    self.logger.log("Releasing slot '%s' to free memory", key,
                                    level=logging.DEBUG)
                    store.release(key)

        run_report = self.logger.finish_run()
        store.set("run_report", run_report)

        return store
