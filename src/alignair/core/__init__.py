"""Shared configurable AlignAIR models and explicit component construction."""
from .config import AlignAIRConfig, GeneSpec
from .components import ModelComponents, register_architecture
from .model import AlignAIR
from .setup import ModelSetup, prepare_model, build_model_config

__all__ = ["AlignAIR", "AlignAIRConfig", "GeneSpec", "ModelComponents", "register_architecture",
           "ModelSetup", "prepare_model", "build_model_config"]
