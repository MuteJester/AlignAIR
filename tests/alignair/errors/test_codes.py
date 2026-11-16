"""Error codes are one scheme, ``ALIGNAIR_<THING>_<STATE>``: every error class of the package has the code this table
pins, and every code but the base's and the CLI boundary's two names a thing and one of seven states. A model that is
not there (a missing file, an id no registry lists, a hub repo or revision that does not exist) is MODEL_NOT_FOUND; a
registry that cannot be reached is REGISTRY_UNAVAILABLE and retryable; a registry that cannot be read is
REGISTRY_INVALID."""
import importlib
import json
import re
import urllib.error
import urllib.request

import pytest

from alignair.errors import (
    AlignAIRError,
    ModelError,
    ModelNotFoundError,
    RegistryError,
    RegistryInvalidError,
    RegistryUnavailableError,
)
from alignair.registry import sources
from alignair.registry.sources import OfflineError

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


def test_a_missing_model_file_is_model_not_found(tmp_path):
    from alignair import Aligner
    with pytest.raises(ModelNotFoundError) as caught:
        Aligner.from_pretrained(str(tmp_path / "missing.alignair"), device="cpu")
    assert caught.value.code == "ALIGNAIR_MODEL_NOT_FOUND" and isinstance(caught.value, ModelError)
    assert caught.value.retryable is False


def _registry(tmp_path, models: dict) -> str:
    (tmp_path / "registry.json").write_text(json.dumps({"models": models}))
    return f"file://{tmp_path}/"


def test_an_id_no_registry_lists_is_model_not_found(tmp_path):
    from alignair.registry.cache import resolve_model
    source = _registry(tmp_path, {"other": {"latest": "1", "versions": {"1": {"file": "o.alignair"}}}})
    with pytest.raises(ModelNotFoundError, match="unknown model 'absent'") as caught:
        resolve_model("absent", sources=[source])
    assert caught.value.context["model_id"] == "absent"


@pytest.mark.parametrize("text", ["{not json", json.dumps({"models": []})], ids=["not JSON", "no models mapping"])
def test_a_registry_that_cannot_be_read_is_invalid_and_not_retryable(tmp_path, text):
    (tmp_path / "registry.json").write_text(text)
    with pytest.raises(RegistryInvalidError) as caught:
        sources.load_registry(f"file://{tmp_path}/")
    assert caught.value.code == "ALIGNAIR_REGISTRY_INVALID" and caught.value.retryable is False
    with pytest.raises(RegistryInvalidError):
        sources.find_model("m", None, [f"file://{tmp_path}/"])


def _unreachable(error):
    def urlopen(url, timeout=None):
        raise error
    return urlopen


@pytest.mark.parametrize("error, retryable", [
    (urllib.error.URLError("no route"), True),
    (TimeoutError("timed out"), True),
    (urllib.error.HTTPError("https://r.example/registry.json", 503, "busy", None, None), True),
    (urllib.error.HTTPError("https://r.example/registry.json", 429, "slow down", None, None), True),
    (urllib.error.HTTPError("https://r.example/registry.json", 404, "not found", None, None), False),
    (urllib.error.HTTPError("https://r.example/registry.json", 403, "forbidden", None, None), False),
], ids=["no route", "timeout", "503", "429", "404", "403"])
def test_a_registry_that_cannot_be_reached_is_unavailable(monkeypatch, error, retryable):
    monkeypatch.setattr(urllib.request, "urlopen", _unreachable(error))
    with pytest.raises(RegistryUnavailableError) as caught:
        sources.load_registry("https://r.example/")
    assert caught.value.code == "ALIGNAIR_REGISTRY_UNAVAILABLE" and caught.value.retryable is retryable
    assert caught.value.__cause__ is error
    with pytest.raises(RegistryUnavailableError) as aggregated:
        sources.find_model("m", None, ["https://r.example/"])
    assert aggregated.value.retryable is retryable


def test_a_local_registry_that_is_not_there_is_unavailable_and_not_retryable(tmp_path):
    with pytest.raises(RegistryUnavailableError) as caught:
        sources.find_model("m", None, [f"file://{tmp_path}/absent/"])
    assert caught.value.retryable is False and isinstance(caught.value.__cause__, FileNotFoundError)


def test_offline_with_the_model_not_cached_is_model_unavailable_and_not_retryable():
    with pytest.raises(OfflineError) as caught:
        sources.find_model("m", None, ["https://r.example/"], offline=True)
    assert caught.value.code == "ALIGNAIR_MODEL_UNAVAILABLE" and caught.value.retryable is False
    assert isinstance(caught.value, RegistryError)


class _Dropping:
    """A remote stream whose connection drops after the first chunk."""

    def __init__(self):
        self.reads = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size=-1):
        self.reads += 1
        if self.reads > 1:
            raise ConnectionResetError("connection reset by peer")
        return b"x" * 16


def test_a_download_whose_connection_drops_is_unavailable_and_retryable(tmp_path, monkeypatch):
    from alignair.registry.cache import download_verified
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None: _Dropping())
    dest = tmp_path / "m" / "1.alignair"
    with pytest.raises(RegistryUnavailableError) as caught:
        download_verified("https://r.example/", "m.alignair", dest, "0" * 64)
    assert caught.value.retryable is True and isinstance(caught.value.__cause__, ConnectionResetError)
    assert not dest.exists()


def test_a_full_disk_during_a_download_stays_an_os_error(tmp_path, monkeypatch):
    from alignair.registry import cache
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None: _Dropping())

    class Full:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write(self, data):
            raise OSError(28, "No space left on device")

    real_open = open
    monkeypatch.setattr("builtins.open", lambda path, mode="r", *a, **k: Full() if str(path).endswith(".part")
                        else real_open(path, mode, *a, **k))
    with pytest.raises(OSError) as caught:
        cache.download_verified("https://r.example/", "m.alignair", tmp_path / "m" / "1.alignair", None)
    assert not isinstance(caught.value, AlignAIRError) and caught.value.errno == 28


def _hub_error(name: str):
    from huggingface_hub import utils
    cls = getattr(utils, name)
    error = cls.__new__(cls)
    Exception.__init__(error, f"{name} from the hub")
    return error


@pytest.mark.parametrize("raised, expected, retryable", [
    (lambda: _hub_error("RepositoryNotFoundError"), ModelNotFoundError, False),
    (lambda: _hub_error("RevisionNotFoundError"), ModelNotFoundError, False),
    (lambda: _hub_error("EntryNotFoundError"), ModelNotFoundError, False),
    (lambda: _hub_error("LocalEntryNotFoundError"), RegistryUnavailableError, True),
    (lambda: ConnectionError("hub unreachable"), RegistryUnavailableError, True),
], ids=["no repo", "no revision", "no file", "unreachable, not cached", "connection"])
def test_a_hub_download_failure_is_not_found_or_unavailable(monkeypatch, raised, expected, retryable):
    pytest.importorskip("huggingface_hub")
    import huggingface_hub

    from alignair.registry import hf
    error = raised()

    def download(**kwargs):
        raise error
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    with pytest.raises(expected) as caught:
        hf.download_from_hub("hf://org/repo")
    assert type(caught.value) is expected and caught.value.retryable is retryable
    assert caught.value.__cause__ is error and caught.value.context["repo_id"] == "org/repo"


def test_a_hub_model_not_cached_offline_is_model_unavailable(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import constants

    from alignair.registry import hf
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "empty-hub"))
    with pytest.raises(OfflineError) as caught:
        hf.download_from_hub("hf://org/absent-repo", offline=True)
    assert caught.value.context["offline"] is True and caught.value.retryable is False
    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_a_hub_model_not_cached_under_the_hubs_own_offline_mode_is_model_unavailable(monkeypatch):
    pytest.importorskip("huggingface_hub")
    import huggingface_hub
    from huggingface_hub import constants

    from alignair.registry import hf
    error = _hub_error("LocalEntryNotFoundError")

    def download(**kwargs):
        raise error
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    with pytest.raises(OfflineError) as caught:
        hf.download_from_hub("hf://org/repo")
    assert caught.value.retryable is False and caught.value.context["offline"] is True


def test_the_cli_reports_a_missing_model_as_not_found(tmp_path, capsys):
    from alignair.cli.main import main
    reads = tmp_path / "reads.fasta"
    reads.write_text(">r\nACGT\n")
    assert main(["--error-format", "json", "predict", "--model", str(tmp_path / "missing.alignair"),
                 "--input", str(reads), "--out", str(tmp_path / "out.tsv")]) == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == "ALIGNAIR_MODEL_NOT_FOUND" and payload["retryable"] is False
