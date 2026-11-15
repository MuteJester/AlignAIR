"""AlignAIR prediction pipeline: model outputs -> finalized AIRR alignments.

An ordered list of named stages over one typed state (``predict.pipeline``). Public entry point ``predict``.
"""
from .config import PredictConfig
from .pipeline import PIPELINE, STAGE_NAMES, Stage, Trace, predict, run_stages
from .state import GeneCall, GermlineAlignment, PredictionState, Predictions, Segments

__all__ = ["predict", "PredictConfig", "GeneCall", "GermlineAlignment", "Predictions", "Segments", "PredictionState",
           "Stage", "PIPELINE", "STAGE_NAMES", "run_stages", "Trace"]
