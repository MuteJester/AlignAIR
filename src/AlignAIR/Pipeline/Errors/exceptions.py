"""Typed exception hierarchy for AlignAIR pipeline errors."""
from typing import List, Optional


class AlignAIRError(Exception):
    """Base exception for all AlignAIR pipeline errors."""
    pass


# -- Fatal errors (pipeline cannot continue) --

class ModelLoadError(AlignAIRError):
    """Model bundle is missing, corrupt, or incompatible."""
    pass


class ConfigError(AlignAIRError):
    """Pipeline configuration is invalid or self-contradictory."""
    pass


class DataConfigError(AlignAIRError):
    """DataConfig pickle is corrupt, missing, or incompatible with model."""
    pass


class StageContractError(AlignAIRError):
    """A stage violated its declared reads/writes contract."""
    pass


class SchemaVersionError(AlignAIRError):
    """Archive or bundle uses a newer schema version than this code supports."""
    pass


# -- Recoverable errors (pipeline can degrade gracefully) --

class SequenceValidationError(AlignAIRError):
    """One or more input sequences are invalid."""

    def __init__(
        self,
        message: str,
        invalid_indices: Optional[List[int]] = None,
        reasons: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.invalid_indices = invalid_indices or []
        self.reasons = reasons or []


class AlleleNotFoundWarning(AlignAIRError):
    """An allele referenced in genotype YAML is not in the model vocabulary."""
    pass


class CheckpointCorruptError(AlignAIRError):
    """A checkpoint file is missing or has a checksum mismatch."""
    pass


class InferenceWarning(AlignAIRError):
    """Non-fatal issue during inference (e.g., NaN in output)."""
    pass
