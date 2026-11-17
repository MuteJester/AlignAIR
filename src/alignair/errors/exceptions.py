"""The domain hierarchy: every error class AlignAIR raises is defined here, so ``alignair.errors`` exports them
all."""
from typing import Any

from .base import AlignAIRError, ExitCode


class InputError(AlignAIRError, ValueError):
    """Invalid user input or sequence data."""
    code = "ALIGNAIR_INPUT_INVALID"
    category = "input"


class UnknownAlleleError(InputError):
    """A genotype names alleles the reference does not hold; their sequences belong in the reference."""
    code = "ALIGNAIR_GENOTYPE_ALLELE_NOT_FOUND"


class DuplicateMetadataIdError(InputError):
    """A metadata join key occurs more than once; neither row may silently win."""
    code = "ALIGNAIR_METADATA_ID_INVALID"


class ConfigurationError(AlignAIRError, ValueError):
    """Invalid configuration or incompatible options."""
    code = "ALIGNAIR_CONFIG_INVALID"
    category = "configuration"


class DataContractError(AlignAIRError, RuntimeError):
    """A record, batch, or component violated its declared data contract."""
    code = "ALIGNAIR_DATA_INVALID"
    category = "data"


class PredictionCountMismatchError(DataContractError):
    """The writer was given a different number of ids / sequences / predictions — a pipeline or model
    defect. Raised instead of silently truncating (and losing reads) via ``zip``."""
    code = "ALIGNAIR_PREDICTION_COUNT_MISMATCH"


class ModelError(AlignAIRError, ValueError):
    """A model artifact is missing, incompatible, or unusable."""
    code = "ALIGNAIR_MODEL_INVALID"
    category = "model"


class ModelFormatError(ModelError):
    """Malformed or unsupported .alignair container."""
    code = "ALIGNAIR_MODEL_FORMAT_INVALID"


class ModelNotFoundError(ModelError):
    """No such model: a model file that does not exist, an id no registry lists, a hub repo, revision or file that is
    not there."""
    code = "ALIGNAIR_MODEL_NOT_FOUND"


class ArtifactIntegrityError(AlignAIRError, RuntimeError):
    """Artifact bytes or bound identities failed verification."""
    code = "ALIGNAIR_ARTIFACT_INTEGRITY_FAILED"
    category = "integrity"


class DownloadIntegrityError(ArtifactIntegrityError):
    """A downloaded artifact failed SHA256 verification."""
    code = "ALIGNAIR_DOWNLOAD_INTEGRITY_FAILED"


class ModelIntegrityError(ModelError, ArtifactIntegrityError):
    """A model section failed verification (also catchable as ValueError)."""
    code = "ALIGNAIR_MODEL_INTEGRITY_FAILED"


class ModelSectionError(ModelError, KeyError):
    """A required model section is absent (also catchable as KeyError)."""
    code = "ALIGNAIR_MODEL_SECTION_MISSING"


class RegistryError(AlignAIRError, RuntimeError, ValueError):
    """Model registry lookup or transfer failed; ``retryable`` says whether a retry may succeed."""
    code = "ALIGNAIR_REGISTRY_FAILED"
    category = "registry"


class RegistryUnavailableError(RegistryError):
    """A registry source or the hub could not be reached or read (the network, a server, a local registry that is not
    there). Retryable unless the server answered that the request itself is wrong or the source is local."""
    code = "ALIGNAIR_REGISTRY_UNAVAILABLE"

    def __init__(self, message: str = "", *, retryable: bool = True, **kwargs: Any) -> None:
        super().__init__(message, retryable=retryable, **kwargs)


class RegistryInvalidError(RegistryError):
    """A registry's catalog cannot be used: not JSON, no models mapping, a malformed entry, a version that fails
    validation."""
    code = "ALIGNAIR_REGISTRY_INVALID"


class OfflineError(RegistryError):
    """A fetch was needed while offline: the model is not available from the cache. Not retryable: the same call
    fails the same way until offline mode is off."""
    code = "ALIGNAIR_MODEL_UNAVAILABLE"


class TrainingError(AlignAIRError, RuntimeError):
    """Training failed."""
    code = "ALIGNAIR_TRAINING_FAILED"
    category = "training"


class TrainingDataError(TrainingError, DataContractError):
    """Simulator or training records lack the fields needed to construct targets."""
    code = "ALIGNAIR_TRAINING_DATA_INVALID"


class TrainingConfigError(ConfigurationError, TrainingError):
    """A training request is invalid (bad hyperparameters / config / reference). Raised *before* the
    full model is allocated so misuse fails fast with an actionable message. Its code is a configuration error's."""


class InferenceError(AlignAIRError, RuntimeError):
    """Prediction or result assembly failed."""
    code = "ALIGNAIR_INFERENCE_FAILED"
    category = "inference"


class AirrAssemblyError(InferenceError):
    """AIRR assembly failed for one record. Carries the record identifier and the originating error so
    an unexpected (non-data) failure surfaces with context instead of silently dropping AIRR fields."""
    code = "ALIGNAIR_AIRR_ASSEMBLY_FAILED"

    def __init__(self, ident: object, original: BaseException) -> None:
        self.ident = ident
        self.original = original
        super().__init__(f"AIRR assembly failed for {ident}: "
                         f"{type(original).__name__}: {original}",
                         context={"stage": "airr_assembly", "record_id": ident})


class NumericalError(AlignAIRError, RuntimeError, ValueError):
    """Non-finite model outputs, losses, or gradients."""
    code = "ALIGNAIR_NUMERICS_INVALID"
    category = "numerical"


class NonFiniteTrainingError(TrainingError, NumericalError):
    """Loss or a gradient became NaN/Inf — training is aborted with a per-task diagnostic rather than
    continuing to corrupt the weights."""
    code = "ALIGNAIR_TRAINING_NUMERICS_INVALID"


class MissingDependencyError(AlignAIRError, ImportError):
    """A requested feature needs an optional dependency."""
    code = "ALIGNAIR_DEPENDENCY_MISSING"
    category = "dependency"


class OperationIOError(AlignAIRError, OSError):
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
