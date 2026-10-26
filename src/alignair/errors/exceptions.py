"""Small domain hierarchy. Specific failures may subclass these in their own modules."""
from .base import AlignAIRError, ExitCode


class InputError(AlignAIRError, ValueError):
    """Invalid user input or sequence data."""
    code = "ALIGNAIR_INPUT_INVALID"
    category = "input"


class ConfigurationError(AlignAIRError, ValueError):
    """Invalid configuration or incompatible options."""
    code = "ALIGNAIR_CONFIG_INVALID"
    category = "configuration"


class DataContractError(AlignAIRError, RuntimeError):
    """A record, batch, or component violated its declared data contract."""
    code = "ALIGNAIR_DATA_CONTRACT"
    category = "data"


class ModelError(AlignAIRError, ValueError):
    """A model artifact is missing, incompatible, or unusable."""
    code = "ALIGNAIR_MODEL_INVALID"
    category = "model"


class ModelFormatError(ModelError):
    """Malformed or unsupported .alignair container."""
    code = "ALIGNAIR_MODEL_FORMAT"


class ArtifactIntegrityError(AlignAIRError, RuntimeError):
    """Artifact bytes or bound identities failed verification."""
    code = "ALIGNAIR_ARTIFACT_INTEGRITY"
    category = "integrity"


class ModelIntegrityError(ModelError, ArtifactIntegrityError):
    """A model section failed verification (also catchable as ValueError)."""
    code = "ALIGNAIR_MODEL_INTEGRITY"


class ModelSectionError(ModelError, KeyError):
    """A required model section is absent (also catchable as KeyError)."""
    code = "ALIGNAIR_MODEL_SECTION_MISSING"


class RegistryError(AlignAIRError, RuntimeError, ValueError):
    """Model registry lookup or transfer failed; retry only when explicitly permitted."""
    code = "ALIGNAIR_REGISTRY_UNAVAILABLE"
    category = "registry"


class TrainingError(AlignAIRError, RuntimeError):
    """Training failed."""
    code = "ALIGNAIR_TRAINING_FAILED"
    category = "training"


class TrainingDataError(TrainingError, DataContractError):
    """Simulator or training records lack the fields needed to construct targets."""
    code = "ALIGNAIR_TRAINING_DATA_INVALID"


class InferenceError(AlignAIRError, RuntimeError):
    """Prediction or result assembly failed."""
    code = "ALIGNAIR_INFERENCE_FAILED"
    category = "inference"


class NumericalError(AlignAIRError, RuntimeError, ValueError):
    """Non-finite model outputs, losses, or gradients."""
    code = "ALIGNAIR_NUMERICAL_FAILURE"
    category = "numerical"


class MissingDependencyError(AlignAIRError, ImportError):
    """A requested feature needs an optional dependency."""
    code = "ALIGNAIR_DEPENDENCY_MISSING"
    category = "dependency"


class AccessPolicyError(AlignAIRError, RuntimeError):
    """A protected dataset or artifact cannot be accessed in this operation."""
    code = "ALIGNAIR_ACCESS_DENIED"
    category = "access"


class OperationIOError(AlignAIRError, RuntimeError):
    """An expected filesystem or operating-system operation failed."""
    code = "ALIGNAIR_IO_FAILED"
    category = "io"


class InternalError(AlignAIRError, RuntimeError):
    """Unexpected defect, classified only at an application boundary."""
    code = "ALIGNAIR_INTERNAL_ERROR"
    category = "internal"
    exit_code = ExitCode.INTERNAL


class OperationCancelledError(AlignAIRError):
    """An application operation was interrupted by the user."""
    code = "ALIGNAIR_INTERRUPTED"
    category = "interrupted"
    exit_code = ExitCode.INTERRUPTED
