import importlib
import struct

import pytest

from alignair.errors import (AlignAIRError, ModelFormatError, ModelIntegrityError, ModelSectionError,
                             RegistryError, TrainingDataError)
from alignair.model_file import container

pytestmark = pytest.mark.component


@pytest.mark.parametrize("module,name,builtin", [
    ("io.sequence_reader", "DuplicateMetadataId", ValueError),
    ("io.airr", "PredictionCountMismatch", RuntimeError),
    ("predict.airr.builder", "AirrAssemblyError", Exception),
    ("registry.cache", "IntegrityError", RuntimeError),
    ("registry.sources", "OfflineError", RuntimeError),
    ("train.guards", "TrainingConfigError", ValueError),
    ("train.guards", "NonFiniteLossError", RuntimeError),
    ("genotype.constraint", "NovelAlleleUnsupportedError", ValueError),
    ("aa.tokenizer", "InvalidResidueError", ValueError),
    ("aa.data", "SeedBudgetExhausted", RuntimeError),
    ("aa.guard.access", "AccessViolation", RuntimeError),
    ("aa.guard.access", "ThresholdArtifactMissing", RuntimeError),
    ("aa.guard.access", "HashMismatch", RuntimeError),
    ("aa.guard.access", "NoAdmissibleThreshold", RuntimeError),
    ("aa.guard.fusion", "FusionContractError", RuntimeError),
    ("aa.guard.tokenizer", "InputTooLong", ValueError),
    ("aa.guard.disjoint", "Gate3DigestMismatch", RuntimeError),
    ("aa.guard.profiles", "NoPublishableProfileSource", Exception),
    ("aa.product.applicability.cohort_loader", "LockedTestError", RuntimeError),
])
def test_existing_exceptions_keep_import_paths_and_builtin_catchability(module, name, builtin):
    cls = getattr(importlib.import_module("alignair." + module), name)
    assert issubclass(cls, AlignAIRError)
    assert issubclass(cls, builtin)
    assert cls.code != AlignAIRError.code


@pytest.mark.parametrize("raw", [b"", b"ALGNAIR", container.MAGIC, container.MAGIC + b"123",
                                 container.MAGIC + struct.pack("<Q", 100) + b"{}",
                                 container.MAGIC + struct.pack("<Q", 1) + b"{",
                                 container.MAGIC + struct.pack("<Q", 2) + b"[]"])
def test_malformed_models_raise_classified_errors_with_paths(tmp_path, raw):
    path = tmp_path / "broken.alignair"
    path.write_bytes(raw)
    with pytest.raises(ModelFormatError) as caught:
        container.read_header(path)
    assert caught.value.context["path"] == str(path)
    assert caught.value.context["stage"] == "model_header"


def test_missing_and_corrupted_sections_have_distinct_errors(tmp_path):
    path = tmp_path / "model.alignair"
    container.write_container(path, {}, {"weights": (b"data", "none")})
    with pytest.raises(ModelSectionError) as caught:
        container.read_section(path, "config")
    assert caught.value.context["section"] == "config"
    path.write_bytes(path.read_bytes()[:-1] + b"!")
    with pytest.raises(ModelIntegrityError) as caught:
        container.read_section(path, "weights")
    assert caught.value.hint


@pytest.mark.parametrize("offset,length", [(-1, 1), (10 ** 100, 1), (0, -1), (0, 10 ** 100)])
def test_invalid_section_bounds_raise_format_errors(tmp_path, offset, length):
    import json
    path = tmp_path / "broken.alignair"
    header = {"sections": {"x": {"offset": offset, "compressed_length": length, "payload_length": 1,
                                  "codec": "none", "compressed_sha256": "hash", "payload_sha256": "hash"}}}
    data = json.dumps(header).encode()
    path.write_bytes(container.MAGIC + struct.pack("<Q", len(data)) + data + b"x")
    with pytest.raises(ModelFormatError):
        container.read_section(path, "x")


def test_invalid_compressed_payload_preserves_decoder_cause():
    with pytest.raises(ModelFormatError) as caught:
        container.decompress(b"invalid-zlib-data", "zlib")
    assert caught.value.__cause__ is not None
    assert caught.value.context["codec"] == "zlib"


def test_registry_failure_is_not_misreported_as_unknown_model(tmp_path, monkeypatch):
    from alignair.registry import cache, sources
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path / "cache"))
    cause = OSError("network unavailable")
    monkeypatch.setattr(sources, "load_registry", lambda *a, **k: (_ for _ in ()).throw(cause))
    with pytest.raises(RegistryError) as caught:
        cache.resolve_model("some-model", sources=["https://example.com/registry"])
    assert caught.value.__cause__ is cause
    assert caught.value.context["model_id"] == "some-model"
    assert "unknown model" not in str(caught.value)


def test_registry_defects_are_not_silently_reclassified(tmp_path, monkeypatch):
    from alignair.registry import cache, sources
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(sources, "load_registry", lambda *a, **k: (_ for _ in ()).throw(AssertionError("bug")))
    with pytest.raises(AssertionError, match="bug"):
        cache.resolve_model("some-model", sources=["file://registry"])


def test_registry_fallback_still_works(tmp_path, monkeypatch):
    from alignair.registry import sources
    registry = {"models": {"igh": {"latest": "1", "versions": {"1": {"file": "m.alignair"}}}}}
    def load(source, **kwargs):
        if source == "bad":
            raise OSError("unavailable")
        return registry
    monkeypatch.setattr(sources, "load_registry", load)
    assert sources.find_model("igh", None, ["bad", "good"])[0] == "good"


def test_training_record_failure_preserves_cause_and_record_context():
    from alignair.train.gym.targets import build_targets
    with pytest.raises(TrainingDataError) as caught:
        build_targets({"sequence_id": "read-42"}, object(), True)
    assert isinstance(caught.value.__cause__, KeyError)
    assert caught.value.context == {"stage": "build_targets", "record_id": "read-42", "field": "sequence"}


def test_missing_gene_coordinates_explain_the_contract_and_keep_operation_context():
    from types import SimpleNamespace
    from alignair.errors import error_context
    from alignair.train.gym.targets import build_targets

    reference = SimpleNamespace(gene=lambda gene: SimpleNamespace(index={"D*01": 0}, sequences=["ACGT"]))
    record = {"sequence": "ACGT", "sequence_id": "read-D", "d_sequence_start": 0,
              "d_sequence_end": 1, "d_call": "D*01"}
    with error_context(step=42, record_index=3):
        with pytest.raises(TrainingDataError) as caught:
            build_targets(record, reference, True)
    assert isinstance(caught.value.__cause__, KeyError)
    assert caught.value.context == {"step": 42, "record_index": 3, "record_id": "read-D",
                                    "stage": "build_targets", "gene": "D",
                                    "fields": ["d_germline_start", "d_germline_end"]}
    assert "ACGT" not in caught.value.to_json()


def test_target_builder_does_not_convert_unexpected_defects_to_invalid_records(monkeypatch):
    from alignair.train.gym import targets
    def broken(*args):
        raise AssertionError("unexpected defect")
    monkeypatch.setattr(targets, "_build_targets", broken)
    with pytest.raises(AssertionError, match="unexpected defect"):
        targets.build_targets({}, object(), True)
