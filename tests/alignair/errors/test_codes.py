"""Error codes are one scheme, ``ALIGNAIR_<THING>_<STATE>``: every error class of the package has the code this table
pins, and every code but the base's and the CLI boundary's two names a thing and one of seven states."""
import importlib
import re

import pytest

from alignair.errors import AlignAIRError

pytestmark = pytest.mark.component

CODES = {
    "alignair.errors.base.AlignAIRError": "ALIGNAIR_ERROR",
    "alignair.errors.exceptions.InputError": "ALIGNAIR_INPUT_INVALID",
    "alignair.errors.exceptions.ConfigurationError": "ALIGNAIR_CONFIG_INVALID",
    "alignair.errors.exceptions.DataContractError": "ALIGNAIR_DATA_INVALID",
    "alignair.errors.exceptions.ModelError": "ALIGNAIR_MODEL_INVALID",
    "alignair.errors.exceptions.ModelFormatError": "ALIGNAIR_MODEL_FORMAT_INVALID",
    "alignair.errors.exceptions.ModelNotFoundError": "ALIGNAIR_MODEL_NOT_FOUND",
    "alignair.errors.exceptions.ArtifactIntegrityError": "ALIGNAIR_ARTIFACT_INTEGRITY_FAILED",
    "alignair.errors.exceptions.ModelIntegrityError": "ALIGNAIR_MODEL_INTEGRITY_FAILED",
    "alignair.errors.exceptions.ModelSectionError": "ALIGNAIR_MODEL_SECTION_MISSING",
    "alignair.errors.exceptions.RegistryError": "ALIGNAIR_REGISTRY_FAILED",
    "alignair.errors.exceptions.RegistryUnavailableError": "ALIGNAIR_REGISTRY_UNAVAILABLE",
    "alignair.errors.exceptions.RegistryInvalidError": "ALIGNAIR_REGISTRY_INVALID",
    "alignair.errors.exceptions.TrainingError": "ALIGNAIR_TRAINING_FAILED",
    "alignair.errors.exceptions.TrainingDataError": "ALIGNAIR_TRAINING_DATA_INVALID",
    "alignair.errors.exceptions.InferenceError": "ALIGNAIR_INFERENCE_FAILED",
    "alignair.errors.exceptions.NumericalError": "ALIGNAIR_NUMERICS_INVALID",
    "alignair.errors.exceptions.MissingDependencyError": "ALIGNAIR_DEPENDENCY_MISSING",
    "alignair.errors.exceptions.OperationIOError": "ALIGNAIR_IO_FAILED",
    "alignair.errors.exceptions.InternalError": "ALIGNAIR_INTERNAL_ERROR",
    "alignair.errors.exceptions.OperationCancelledError": "ALIGNAIR_INTERRUPTED",
    "alignair.registry.sources.OfflineError": "ALIGNAIR_MODEL_UNAVAILABLE",
    "alignair.registry.cache.DownloadIntegrityError": "ALIGNAIR_DOWNLOAD_INTEGRITY_FAILED",
    "alignair.genotype.files.UnknownAlleleError": "ALIGNAIR_GENOTYPE_ALLELE_NOT_FOUND",
    "alignair.io.airr.PredictionCountMismatchError": "ALIGNAIR_PREDICTION_COUNT_MISMATCH",
    "alignair.io.metadata.DuplicateMetadataIdError": "ALIGNAIR_METADATA_ID_INVALID",
    "alignair.train.guards.TrainingConfigError": "ALIGNAIR_CONFIG_INVALID",
    "alignair.train.guards.NonFiniteTrainingError": "ALIGNAIR_TRAINING_NUMERICS_INVALID",
    "alignair.predict.airr.builder.AirrAssemblyError": "ALIGNAIR_AIRR_ASSEMBLY_FAILED",
}
STATES = ("INVALID", "MISSING", "FAILED", "UNAVAILABLE", "MISMATCH", "NOT_FOUND", "INTERRUPTED")
UNSCHEMED = {"ALIGNAIR_ERROR", "ALIGNAIR_INTERNAL_ERROR", "ALIGNAIR_INTERRUPTED"}
RETRYABLE = {"alignair.errors.exceptions.RegistryUnavailableError"}


def _class(path: str):
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


def _package_classes() -> set[str]:
    for path in CODES:
        _class(path)
    seen, todo = set(), [AlignAIRError]
    while todo:
        cls = todo.pop()
        if cls.__module__.split(".")[0] == "alignair":
            seen.add(f"{cls.__module__}.{cls.__qualname__}")
        todo.extend(cls.__subclasses__())
    return seen


@pytest.mark.parametrize("path", list(CODES))
def test_every_error_class_has_its_code_and_retry_default(path):
    cls = _class(path)
    error = cls("r1", ValueError("x")) if cls.__name__ == "AirrAssemblyError" else cls("x")
    assert error.code == CODES[path]
    assert error.retryable is (path in RETRYABLE)
    assert error.to_dict()["code"] == CODES[path]


def test_the_table_names_every_error_class_of_the_package():
    assert _package_classes() == set(CODES)


def test_every_code_names_a_thing_and_one_of_seven_states():
    scheme = re.compile(rf"^ALIGNAIR_[A-Z]+(_[A-Z]+)*?_({'|'.join(STATES)})$")
    assert [code for code in set(CODES.values()) - UNSCHEMED if not scheme.match(code)] == []
