"""AlignAIR prediction pipeline: model outputs -> finalized AIRR alignments.

An ordered list of named stages over one typed state (``predict.pipeline``). Public entry point ``predict``.
"""
from .config import PredictConfig
from .pipeline import PIPELINE, STAGE_NAMES, Stage, Trace, predict, run_stages
from .state import GeneCall, GermlineAlignment, Predictions, PredictionState, Segments

__all__ = [
    "PIPELINE",
    "STAGE_NAMES",
    "GeneCall",
    "GermlineAlignment",
    "PredictConfig",
    "PredictionState",
    "Predictions",
    "Segments",
    "Stage",
    "Trace",
    "predict",
    "run_stages",
]
