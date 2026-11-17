"""One set of CLI options (P2-D9): ``--device`` defaults to auto with one help text; every command that resolves a
model takes the same model-source options and passes them on; machine output is ``--json`` everywhere (``--format``
is gone)."""
import argparse
import json

import pytest

from alignair.cli.main import build_parser, main

pytestmark = pytest.mark.component

DEVICE = (("predict",), ("demo",), ("benchmark",), ("train",))
RESOLVERS = (("predict",), ("benchmark",), ("models", "get"), ("models", "path"))
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
    ["analyze", "rows.tsv", "--format", "json"],
    ["benchmark", "--model", "m.alignair", "--format", "json"],
], ids=["analyze --format", "benchmark --format"])
def test_the_removed_spellings_are_refused(argv, capsys):
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args(argv)
    assert caught.value.code == 2


@pytest.mark.parametrize("argv", [
    ["predict", "--model", "hf://org/repo", "--input", "reads.fasta", "--out", "out.tsv", "--quiet"],
    ["benchmark", "--model", "hf://org/repo", "--dataconfig", "HUMAN_IGH_OGRDB"],
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
