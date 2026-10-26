"""Shared exception and diagnostics API. Implementation uses only Python's standard library.

Use domain exceptions at validation/I/O boundaries, ``raise ... from cause`` when
translating a failure, and ``error_context`` for operation metadata. The library
never configures logging, installs exception hooks, or retries automatically.
"""
from .base import AlignAIRError, ExitCode
from .context import current_error_context, error_context
from .exceptions import (
    AccessPolicyError, ArtifactIntegrityError, ConfigurationError, DataContractError,
    InferenceError, InputError, InternalError, MissingDependencyError, ModelError,
    ModelFormatError, ModelIntegrityError, ModelSectionError, NumericalError,
    OperationCancelledError, OperationIOError, RegistryError, TrainingDataError, TrainingError,
)
from .reporting import error_reporting, format_error, log_error, report_error

__all__ = [
    "AlignAIRError", "ExitCode", "AccessPolicyError", "ArtifactIntegrityError",
    "ConfigurationError", "DataContractError", "InferenceError", "InputError", "InternalError",
    "MissingDependencyError", "ModelError", "ModelFormatError", "ModelIntegrityError",
    "ModelSectionError", "NumericalError", "OperationCancelledError", "OperationIOError", "RegistryError",
    "TrainingDataError", "TrainingError", "current_error_context", "error_context",
    "error_reporting", "format_error", "log_error", "report_error",
]
