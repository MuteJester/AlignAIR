"""AlignAIR prediction pipeline: model outputs -> finalized AIRR alignments.

Functional typed pipeline (pure stages over immutable dataclasses). Public entry point ``predict``.
"""
from .config import PredictConfig
from .pipeline import predict
from .state import GeneCall, GermlineAlignment, Predictions, Segments
from .forward import forward_batch, prediction_batches
from .probabilities import class_probabilities
from .features import component_features

__all__ = ["predict", "PredictConfig", "GeneCall", "GermlineAlignment", "Predictions", "Segments",
           "forward_batch", "prediction_batches", "class_probabilities", "component_features"]
