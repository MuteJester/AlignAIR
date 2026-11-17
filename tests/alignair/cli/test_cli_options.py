"""One set of CLI options (P2-D9): ``--device`` defaults to auto with one help text; every command that resolves a
model takes the same model-source options and passes them on; machine output is ``--json`` everywhere (``--format``
is gone); ``info`` takes a path or a registry id (``models info`` and the ``model`` alias are gone); ``genotype``
reads its table as a positional input (``--airr`` is gone); a disk error while caching a model is IO_FAILED."""
import argparse
import hashlib
import json
import shutil

import pytest

from alignair.cli.main import build_parser, main
from tests.alignair.predict.golden_rows import TEST_CHIP

pytestmark = pytest.mark.component

DEVICE = (("predict",), ("demo",), ("benchmark",), ("train",))
RESOLVERS = (("predict",), ("benchmark",), ("info",), ("models", "get"), ("models", "path"))
CATALOG = (("models", "list"), ("models", "update"), ("models", "verify"))
SOURCE = ("--registry", "--offline", "--revision", "--hf-token")
JSON = (("analyze",), ("benchmark",), ("compare",), ("info",), ("doctor",), ("reference", "list"))


def _parser(*path):
    parser = build_parser()
    for name in path:
        (sub,) = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
        parser = sub.choices[name]
    return parser


def _option(path, flag):
    found = [a for a in _parser(*path)._actions if flag in a.option_strings]
    return found[0] if found else None


def test_every_device_option_defaults_to_auto_with_one_help_text():
    options = [_option(path, "--device") for path in DEVICE]
    assert [o.default for o in options] == ["auto"] * len(DEVICE)
    assert len({o.help for o in options}) == 1
    assert "default: auto" in options[0].help and "MPS" in options[0].help


@pytest.mark.parametrize("path", RESOLVERS, ids=" ".join)
def test_every_command_that_resolves_a_model_takes_the_model_source_options(path):
    assert [flag for flag in SOURCE if _option(path, flag) is None] == []


@pytest.mark.parametrize("path", CATALOG, ids=" ".join)
def test_the_catalog_commands_take_the_registry_options(path):
    assert _option(path, "--registry") is not None and _option(path, "--offline") is not None


def test_each_model_source_option_has_one_help_text():
    for flag in SOURCE:
        paths = RESOLVERS + (CATALOG if flag in ("--registry", "--offline") else ())
        assert len({_option(path, flag).help for path in paths}) == 1, flag


@pytest.mark.parametrize("path", JSON, ids=" ".join)
def test_machine_output_is_json_everywhere(path):
    assert _option(path, "--json") is not None and _option(path, "--format") is None


@pytest.mark.parametrize("argv", [
    ["models", "info", "some-model"],
    ["model", "list"],
    ["analyze", "rows.tsv", "--format", "json"],
    ["benchmark", "--model", "m.alignair", "--format", "json"],
    ["genotype", "--airr", "rows.tsv", "--out", "g.json"],
], ids=["models info", "model alias", "analyze --format", "benchmark --format", "genotype --airr"])
def test_the_removed_spellings_are_refused(argv, capsys):
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args(argv)
    assert caught.value.code == 2


@pytest.mark.parametrize("argv", [
    ["predict", "--model", "hf://org/repo", "--input", "reads.fasta", "--out", "out.tsv", "--quiet"],
    ["benchmark", "--model", "hf://org/repo", "--dataconfig", "HUMAN_IGH_OGRDB"],
    ["info", "hf://org/repo"],
    ["models", "get", "hf://org/repo"],
    ["models", "path", "hf://org/repo"],
], ids=lambda argv: " ".join(argv[:2]) if argv[0] == "models" else argv[0])
def test_every_command_that_resolves_a_model_passes_the_revision_and_token_on(argv, monkeypatch, tmp_path, capsys):
    pytest.importorskip("huggingface_hub")
    import huggingface_hub
    from huggingface_hub import constants
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.chdir(tmp_path)
    seen = {}

    def download(**kwargs):
        seen.update(kwargs)
        raise ConnectionError("stop here")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert main(["--error-format", "json", *argv, "--revision", "v1", "--hf-token", "secret"]) != 0
    assert seen["repo_id"] == "org/repo" and seen["revision"] == "v1" and seen["token"] == "secret"
    assert json.loads(capsys.readouterr().err)["code"] == "ALIGNAIR_REGISTRY_UNAVAILABLE"


def _chip_registry(tmp_path, monkeypatch) -> str:
    """A local registry listing the test chip as ``chip@1.0.0``, and an empty model cache; returns its url."""
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("ALIGNAIR_REGISTRY", raising=False)
    registry = tmp_path / "registry"
    (registry / "chip").mkdir(parents=True)
    shutil.copyfile(TEST_CHIP, registry / "chip" / "1.0.0.alignair")
    sha = hashlib.sha256(TEST_CHIP.read_bytes()).hexdigest()
    (registry / "registry.json").write_text(json.dumps({"schema": "alignair.registry.v1", "models": {"chip": {
        "latest": "1.0.0", "versions": {"1.0.0": {"file": "chip/1.0.0.alignair", "artifact_sha256": sha}}}}}))
    return f"file://{registry}"


def test_info_takes_a_registry_id_and_prints_the_card_of_its_file(tmp_path, monkeypatch, capsys):
    registry = _chip_registry(tmp_path, monkeypatch)
    assert main(["info", str(TEST_CHIP)]) == 0
    card = capsys.readouterr().out
    assert main(["info", "chip@1.0.0", "--registry", registry]) == 0
    assert capsys.readouterr().out == card and "alleles:" in card


@pytest.mark.parametrize("argv", [
    ["info", "chip@1.0.0"],
    ["predict", "--model", "chip@1.0.0", "--input", "reads.fasta", "--out", "out.tsv", "--quiet"],
    ["benchmark", "--model", "chip@1.0.0", "--dataconfig", "HUMAN_IGH_OGRDB"],
], ids=lambda argv: argv[0])
def test_a_disk_error_while_caching_a_model_is_io_failed(argv, tmp_path, monkeypatch, capsys):
    from alignair.registry import cache
    registry = _chip_registry(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)

    def full(*args, **kwargs):
        raise OSError(28, "No space left on device", str(tmp_path / "cache"))
    monkeypatch.setattr(cache, "download_verified", full)
    assert main(["--error-format", "json", *argv, "--registry", registry]) != 0
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == "ALIGNAIR_IO_FAILED" and payload["context"]["errno"] == 28


def test_info_on_a_model_file_that_does_not_exist_is_model_not_found(tmp_path, capsys):
    assert main(["--error-format", "json", "info", str(tmp_path / "missing.alignair")]) != 0
    assert json.loads(capsys.readouterr().err)["code"] == "ALIGNAIR_MODEL_NOT_FOUND"

