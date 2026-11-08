"""The prediction pipeline: model + reference + config -> AIRR-contract records, as an ordered list of named stages.

``PIPELINE`` is the stages in their order, each a function of ``predict.stages`` over one ``PredictionState`` that sets
the fields it names. ``run_stages`` runs them with hooks: a hook wraps every stage call, ``hook(stage, state, run)``,
the first hook outermost, and may replace what the stage returns. ``Trace`` is a hook that captures each stage's output
state and its seconds. ``predict`` is the pipeline run without hooks.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..errors import InferenceError
from . import stages
from .config import PredictConfig
from .state import PredictionState


@dataclass(frozen=True)
class Stage:
    """A named step of the pipeline: ``run`` and the state fields it sets."""
    name: str
    run: Callable[[PredictionState], PredictionState]
    sets: tuple[str, ...]


StageHook = Callable[[Stage, PredictionState, Callable[[PredictionState], PredictionState]], PredictionState]

PIPELINE: tuple[Stage, ...] = (
    Stage("input_policy", stages.prepare_reads, ("reads", "cropped")),
    Stage("forward", stages.run_forward, ("batches",)),
    Stage("clean", stages.decode_outputs, ("predictions",)),
    Stage("canonicalize", stages.canonicalize_reads, ("canonical",)),
    Stage("segments", stages.correct_spans, ("segments",)),
    Stage("alleles", stages.select_calls, ("calls",)),
    Stage("germline", stages.align_reads, ("alignments",)),
    Stage("better_v", stages.promote_better_v, ("alignments",)),
    Stage("v_presence", stages.find_absent_v, ("no_v",)),
    Stage("ties", stages.list_v_ties, ("v_ties",)),
    Stage("records", stages.build_records, ("records",)),
    Stage("airr", stages.build_rows, ("rows",)),
)
STAGE_NAMES: tuple[str, ...] = tuple(stage.name for stage in PIPELINE)


def _wrapped(stage: Stage, hook: StageHook, run):
    return lambda state: hook(stage, state, run)


def run_stages(state: PredictionState, stages: Sequence[Stage] = PIPELINE,
               hooks: Sequence[StageHook] = ()) -> PredictionState:
    """Run ``stages`` in order on ``state``, each call wrapped by ``hooks`` (the first outermost). InferenceError when
    a stage leaves a field it sets unset."""
    for stage in stages:
        run = stage.run
        for hook in reversed(hooks):
            run = _wrapped(stage, hook, run)
        state = run(state)
        for field in stage.sets:
            if getattr(state, field) is None:
                raise InferenceError(f"stage {stage.name} left {field} unset",
                                     context={"stage": stage.name, "field": field})
    return state


class Trace:
    """A hook that keeps each stage's output state (``states``) and its seconds (``seconds``), in run order."""

    def __init__(self):
        self.states: dict[str, PredictionState] = {}
        self.seconds: dict[str, float] = {}

    def __call__(self, stage: Stage, state: PredictionState, run) -> PredictionState:
        start = time.perf_counter()
        out = run(state)
        self.seconds[stage.name] = time.perf_counter() - start
        self.states[stage.name] = out
        return out


def predict(model, sequences, reference, cfg: PredictConfig, device: str = "cpu"):
    """The AIRR-contract records of ``sequences``: the pipeline run without hooks."""
    return run_stages(PredictionState.start(model, reference, cfg, device, sequences)).rows
