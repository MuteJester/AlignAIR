"""What the package exports (P2-D10). The top level is the supported API; every error class lives in and is exported by
``alignair.errors``; ``alignair.io`` and ``alignair.registry`` export what a user calls (``load_metadata`` is gone,
U-06; the publish tools stay in ``alignair.registry.publish``); a registry list is ``registry=`` everywhere, and one
registry may be given as a string; the public API is type-hinted."""
import hashlib
import importlib
import inspect
import json
import pkgutil
from pathlib import Path

import pytest

import alignair
from tests.support.public_api import public_api

pytestmark = pytest.mark.component

TOP_LEVEL = {"AirrWriter", "AlignAIRError", "Aligner", "PredictionResult", "ReferenceSet", "TrainingConfig",
             "TrainingRun", "__version__", "compare_airr", "iter_sequences", "read_sequences", "run_training",
             "write_airr"}
IO = {"MetadataIndex", "SequenceReadBatch", "build_metadata_index", "iter_sequence_batches", "lookup_metadata_rows"}
REGISTRY = {"DEFAULT_REGISTRY", "download_from_hub", "find_model", "is_hf_repo_spec", "load_registry",
            "maybe_notify_updates", "parse_hf_spec", "resolve_model", "resolve_sources", "validate_registry"}


def test_the_top_level_exports_the_supported_api():
    assert set(alignair.__all__) == TOP_LEVEL
    assert all(hasattr(alignair, name) for name in TOP_LEVEL)


@pytest.mark.parametrize("name, module", [
    ("AlignAIR", "alignair.core"), ("AlignAIRConfig", "alignair.core.config"), ("PredictConfig", "alignair.predict"),
    ("resolve_device", "alignair.aligner")])
def test_an_internal_name_left_the_top_level_and_stays_importable_from_its_module(name, module):
    assert not hasattr(alignair, name)
    assert hasattr(importlib.import_module(module), name)


def _package_error_classes() -> set[type]:
    """Every AlignAIRError subclass the package defines (all of its modules imported)."""
    for module in pkgutil.walk_packages(alignair.__path__, "alignair."):
        importlib.import_module(module.name)
    found, todo = set(), [alignair.AlignAIRError]
    while todo:
        cls = todo.pop()
        if cls.__module__.split(".")[0] == "alignair":
            found.add(cls)
        todo.extend(cls.__subclasses__())
    return found


def test_alignair_errors_defines_and_exports_every_error_class():
    from alignair import errors
    classes = _package_error_classes()
    assert len(classes) > 20
    assert sorted(c.__qualname__ for c in classes if not c.__module__.startswith("alignair.errors.")) == []
    assert sorted(c.__name__ for c in classes if getattr(errors, c.__name__, None) is not c
                  or c.__name__ not in errors.__all__) == []


def test_an_error_class_is_exported_from_alignair_errors_only():
    from alignair import io, registry
    moved = {"DuplicateMetadataIdError", "PredictionCountMismatchError", "DownloadIntegrityError", "OfflineError"}
    for package in (alignair, io, registry):
        assert not moved & set(package.__all__) and not [name for name in moved if hasattr(package, name)]


def test_alignair_io_exports_what_the_product_uses_and_load_metadata_is_gone():
    from alignair import io
    from alignair.io import metadata
    assert set(io.__all__) == IO
    assert not hasattr(io, "load_metadata") and not hasattr(metadata, "load_metadata")


def test_alignair_registry_exports_user_calls_and_the_publish_tools_stay_in_publish():
    from alignair import registry
    from alignair.registry import publish
    assert set(registry.__all__) == REGISTRY
    assert not [name for name in ("publish_bundle", "publish_local", "upload_version") if hasattr(registry, name)]
    assert callable(publish.publish_bundle) and callable(publish.publish_local) and callable(publish.upload_version)


def _registry_functions():
    from alignair import Aligner
    from alignair.registry import cache, find_model, maybe_notify_updates, resolve_model, resolve_sources
    return [resolve_model, find_model, maybe_notify_updates, resolve_sources, cache.verify_installed,
            Aligner.from_pretrained]


@pytest.mark.parametrize("function", _registry_functions(), ids=lambda f: f.__qualname__)
def test_a_registry_list_is_called_registry_everywhere(function):
    parameters = inspect.signature(function).parameters
    assert "registry" in parameters and not {"sources", "sources_list", "srcs", "cli"} & set(parameters)


def _one_model_registry(tmp_path, monkeypatch) -> str:
    """A local registry listing one artifact as ``m@1.0.0``, with an empty model cache; returns its url."""
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("ALIGNAIR_REGISTRY", raising=False)
    registry = tmp_path / "registry"
    (registry / "m").mkdir(parents=True)
    (registry / "m" / "1.0.0.alignair").write_bytes(b"model bytes")
    sha = hashlib.sha256(b"model bytes").hexdigest()
    (registry / "registry.json").write_text(json.dumps({"schema": "alignair.registry.v1", "models": {"m": {
        "latest": "1.0.0", "versions": {"1.0.0": {"file": "m/1.0.0.alignair", "artifact_sha256": sha}}}}}))
    return f"file://{registry}"


def test_one_registry_may_be_given_as_a_string(tmp_path, monkeypatch):
    from alignair.registry import find_model, resolve_model, resolve_sources
    url = _one_model_registry(tmp_path, monkeypatch)
    assert resolve_sources(url) == [url]
    assert find_model("m", None, url)[0] == url
    path = resolve_model("m@1.0.0", registry=url)
    assert path.read_bytes() == b"model bytes" and resolve_model("m@1.0.0", registry=[url]) == path


def test_from_pretrained_looks_an_id_up_in_one_registry_given_as_a_string(tmp_path, monkeypatch):
    import shutil

    from alignair import Aligner
    from tests.alignair.predict.golden_rows import TEST_CHIP
    url = _one_model_registry(tmp_path, monkeypatch)
    registry = tmp_path / "registry"
    shutil.copyfile(TEST_CHIP, registry / "m" / "1.0.0.alignair")
    sha = hashlib.sha256(TEST_CHIP.read_bytes()).hexdigest()
    catalog = json.loads((registry / "registry.json").read_text())
    catalog["models"]["m"]["versions"]["1.0.0"]["artifact_sha256"] = sha
    (registry / "registry.json").write_text(json.dumps(catalog))
    aligner = Aligner.from_pretrained("m@1.0.0", registry=url, device="cpu")
    assert hashlib.sha256(Path(aligner.model_path).read_bytes()).hexdigest() == sha


def _unhinted(function) -> list[str]:
    """The parameters of ``function`` without a type hint, and ``return`` when its return has none (``__init__``
    aside)."""
    signature = inspect.signature(inspect.unwrap(function))
    missing = [name for name, parameter in signature.parameters.items()
               if name not in ("self", "cls") and parameter.annotation is inspect.Parameter.empty]
    if signature.return_annotation is inspect.Signature.empty and function.__name__ != "__init__":
        missing.append("return")
    return missing


def test_the_public_api_is_type_hinted():
    missing = {name: _unhinted(value) for name, value in public_api() if inspect.isfunction(value)}
    assert {name: gaps for name, gaps in missing.items() if gaps} == {}


def test_alignair_train_exports_the_run_and_its_settings_and_not_the_engine():
    """The engine and the loop are internal (``alignair.train.engine``, ``.loop``); a run's callback receives a
    ``TrainingIteration`` (its ``result`` a ``StepResult``), so those two stay."""
    from alignair import train
    assert set(train.__all__) == {"CosineSchedule", "NovelAlleles", "StepResult", "TrainingConfig",
                                  "TrainingIteration", "TrainingRun", "run_training"}
    assert not [name for name in ("TrainingEngine", "ValidationResult", "build_training_engine",
                                  "training_iterations") if hasattr(train, name)]


def test_run_trainings_callback_is_hinted_with_what_it_receives():
    from alignair import run_training
    assert "TrainingIteration" in str(inspect.signature(run_training).parameters["on_iteration"].annotation)


@pytest.mark.parametrize("function, parameter", [
    ("alignair.reference.reference_set:ReferenceSet.from_dataconfigs", "dataconfigs"),
    ("alignair.evaluate.benchmark:generate_labeled", "dataconfig"),
    ("alignair.evaluate.benchmark:run_benchmark", "dataconfig")])
def test_a_function_that_needs_a_dataconfig_object_is_not_hinted_to_take_a_name(function, parameter):
    """These read the dataconfig's own fields: a name (``"HUMAN_IGH_OGRDB"``) fails there, so the hint says
    ``DataConfig``."""
    module, _, qualname = function.partition(":")
    value = importlib.import_module(module)
    for part in qualname.split("."):
        value = getattr(value, part)
    assert "str" not in str(inspect.signature(value).parameters[parameter].annotation)
