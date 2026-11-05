"""AlignAIR prediction pipeline: model outputs -> finalized AIRR alignments.

Functional typed pipeline (pure stages over immutable dataclasses). Public entry point ``predict``.
"""
from .config import PredictConfig
from .pipeline import predict
from .state import GeneCall, GermlineAlignment, Predictions, Segments
from .forward import collect_prediction_outputs, forward_batch, prediction_batches

__all__ = ["predict", "PredictConfig", "GeneCall", "GermlineAlignment", "Predictions", "Segments",
           "forward_batch", "prediction_batches", "collect_prediction_outputs"]
