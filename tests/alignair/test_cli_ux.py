"""CLI UX: --json / --verbose / completion, and the dependency-story alignment (#93)."""
import json
import tomllib
from pathlib import Path

import pytest

from alignair import cli


def _run(capsys, argv):
    code = 0
    try:
        cli.main(argv)
    except SystemExit as e:
        code = e.code or 0
    return capsys.readouterr().out, code


def test_doctor_json_is_machine_readable(capsys):
    out, code = _run(capsys, ["doctor", "--json"])
    rep = json.loads(out)
    assert "ok" in rep and rep["core"]["torch"]["present"] is True
    assert "GenAIRR" in rep["core"] and "parasail" in rep["optional"]
    assert code == 0


def test_doctor_verbose_adds_platform(capsys):
    out, _ = _run(capsys, ["doctor", "--json", "--verbose"])
    assert "platform" in json.loads(out)


def test_model_list_json(capsys):
    from alignair.hub import MODEL_CATALOG
    out, _ = _run(capsys, ["model", "list", "--json"])
    assert json.loads(out) == MODEL_CATALOG


def test_model_inspect_json(capsys, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("GenAIRR")
    import torch
    from alignair.serialization.dnalignair_bundle import save_dnalignair_bundle
    from alignair.config.dnalignair_config import DNAlignAIRConfig
    from alignair.core.dnalignair import DNAlignAIR
    torch.manual_seed(0)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64))
    bundle = tmp_path / "b"
    save_dnalignair_bundle(bundle, model=model, dataconfigs=["HUMAN_IGH_OGRDB"], locus="IGH")
    out, _ = _run(capsys, ["model", "inspect", str(bundle), "--json"])
    info = json.loads(out)
    assert info["locus"] == "IGH" and info["config"]["d_model"] == 32
    assert info["reference"]["dataconfigs"] == ["HUMAN_IGH_OGRDB"]


def test_completion_prints_activation_line(capsys):
    out, _ = _run(capsys, ["completion", "bash"])
    assert "register-python-argcomplete alignair" in out
    out_zsh, _ = _run(capsys, ["completion", "zsh"])
    assert "bashcompinit" in out_zsh


def test_cli_extra_dependency_story():
    """The advertised [cli] extra matches reality: argcomplete in, dead typer/rich out."""
    pp = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    cli_extra = " ".join(pp["project"]["optional-dependencies"]["cli"])
    assert "argcomplete" in cli_extra
    assert "typer" not in cli_extra and "rich" not in cli_extra
