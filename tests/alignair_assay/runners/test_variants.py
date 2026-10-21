"""Checkpoint variants: one AlignAIR checkpoint run as its own tool, ``<base>@<label>``.

The tool id is parsed and refused at parse time; the variant takes its checkpoint from ``RunnerOptions.checkpoints``
by label, and its fingerprint, run records and paths carry the full id. The plain tools keep every record they had:
``fixtures/plain_tool_records.json`` was written before variants existed, and the plain fingerprints, the ``all``
identity, the paths and the real run identity (all but its code digest) must still equal it.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from alignair_assay.cli import main
from alignair_assay.cli.common import checkpoint_map, checkpoint_spec
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import sha256_file
from alignair_assay.core.records import digest
from alignair_assay.runners import RUNNERS, VARIANTS, make_runner, parse_tool
from alignair_assay.runners import alignair as alignair_module
from alignair_assay.runners import alignair_api as api_module
from alignair_assay.runners.alignair import AlignAIRLocal
from alignair_assay.runners.alignair_api import AlignAIRApi
from alignair_assay.runners.base import RunnerOptions
from alignair_assay.workflow import real_steps
from alignair_assay.workflow.orchestrate import RunConfig, identity
from alignair_assay.workflow.real import RealConfig, manifest_identity

RECORD = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "plain_tool_records.json").read_text())


@pytest.fixture()
def fixed_code(tmp_path, monkeypatch):
    """The code a local checkpoint runs and the API driver, fixed as the stored record has them."""
    monkeypatch.setattr(alignair_module, "local_code_identity", lambda: dict(RECORD["code_identity"]))
    driver = tmp_path / "api_predict.py"
    driver.write_text(RECORD["driver_bytes"])
    monkeypatch.setattr(api_module, "DRIVER", driver)


def _checkpoint(path: Path, content: bytes = b"weights") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# --- parsing ---------------------------------------------------------------------------------------------------

def test_parse_tool_splits_at_the_one_at_sign():
    assert parse_tool("igblast") == ("igblast", None)
    assert parse_tool("alignair-api") == ("alignair-api", None)
    assert parse_tool("alignair-api@cand") == ("alignair-api", "cand")
    assert parse_tool("alignair-local@v1.2_rc-3") == ("alignair-local", "v1.2_rc-3")
    assert parse_tool("alignair-local@" + "x" * 32) == ("alignair-local", "x" * 32)


@pytest.mark.parametrize("label", ["", "a/b", "a\\b", "a b", "x" * 33, "a@b", "é", "a:b", "a*b", "a\nb"])
def test_a_bad_label_is_refused_at_parse_time(label):
    with pytest.raises(ValueError, match="label"):
        parse_tool(f"alignair-api@{label}")


@pytest.mark.parametrize("name", ["igblast@x", "alignair@x", "oracle@x", "nope@x"])
def test_only_the_checkpoint_tools_take_a_label(name):
    with pytest.raises(ValueError, match="alignair-local and alignair-api"):
        parse_tool(name)


# --- building a variant ------------------------------------------------------------------------------------------

def test_make_runner_builds_the_variant_with_its_labels_checkpoint(tmp_path):
    base = _checkpoint(tmp_path / "base.alignair", b"base")
    cand = _checkpoint(tmp_path / "cand.alignair", b"cand")
    options = RunnerOptions(alignair_checkpoint=str(tmp_path / "plain.alignair"), device="cuda", batch_size=16,
                            threads=3, cpus="0-2", checkpoints={"base": str(base), "cand": str(cand)})
    api = make_runner("alignair-api@cand", options)
    assert isinstance(api, AlignAIRApi) and api.name == "alignair-api@cand" and api.label == "cand"
    assert (api.checkpoint, api.device, api.batch_size, api.threads, api.cpus) == (cand, "cuda", 16, 3, "0-2")
    local = make_runner("alignair-local@base", options)
    assert isinstance(local, AlignAIRLocal) and not isinstance(local, AlignAIRApi)
    assert local.name == "alignair-local@base" and local.checkpoint == base
    # the class keeps its plain name: only the instance is the variant
    assert AlignAIRApi.name == "alignair-api" and AlignAIRLocal.name == "alignair-local"
    assert set(VARIANTS) == {"alignair-local", "alignair-api"} and "alignair-api@cand" not in RUNNERS


def test_an_unknown_label_names_the_known_ones(tmp_path):
    options = RunnerOptions(checkpoints={"b": str(_checkpoint(tmp_path / "b.alignair")),
                                         "a": str(_checkpoint(tmp_path / "a.alignair"))})
    with pytest.raises(ValueError, match=r"unknown checkpoint label 'c' \(alignair-api@c\); known labels: a, b"):
        make_runner("alignair-api@c", options)
    with pytest.raises(ValueError, match=r"known labels: none"):
        make_runner("alignair-api@c", RunnerOptions())


def test_a_missing_checkpoint_is_refused_before_any_work(tmp_path):
    options = RunnerOptions(checkpoints={"a": str(tmp_path / "missing.alignair")})
    with pytest.raises(FileNotFoundError, match=r"alignair-api@a.*missing\.alignair"):
        make_runner("alignair-api@a", options)


def test_checkpoint_paths_are_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    options = RunnerOptions(checkpoints={"a": "models/a.alignair", "b": "~/b.alignair"})
    assert options.checkpoints == {"a": str(tmp_path / "models" / "a.alignair"), "b": str(Path.home() / "b.alignair")}
    assert RunnerOptions().checkpoints == {}
    hash(options)  # the options stay hashable


def test_the_variants_fingerprint_adds_its_label_to_the_checkpoints_digest(tmp_path, fixed_code):
    checkpoint = _checkpoint(tmp_path / "model.alignair")
    options = RunnerOptions(alignair_checkpoint=str(checkpoint), checkpoints={"a": str(checkpoint)})
    for base in ("alignair-local", "alignair-api"):
        plain = make_runner(base, options).fingerprint()
        variant = make_runner(f"{base}@a", options).fingerprint()
        assert variant == {**plain, "label": "a"}
        assert variant["checkpoint_sha256"] == sha256_file(checkpoint)
    other = _checkpoint(tmp_path / "other.alignair", b"other")
    changed = make_runner("alignair-api@a", RunnerOptions(checkpoints={"a": str(other)})).fingerprint()
    assert changed["checkpoint_sha256"] == sha256_file(other) != sha256_file(checkpoint)
    # the same checkpoint under two labels is two tools
    two = RunnerOptions(checkpoints={"a": str(checkpoint), "b": str(checkpoint)})
    assert make_runner("alignair-api@a", two).fingerprint() != make_runner("alignair-api@b", two).fingerprint()


def test_a_variants_run_record_names_the_full_id(reference, tmp_path, monkeypatch):
    def fake_timed_command(argv, *, log_path, env=None, cpus=None):
        Path(argv[argv.index("--out") + 1]).write_text("sequence_id\nr1\n")
        Path(log_path).write_text("")
        return 0, 0.1, None

    monkeypatch.setattr(api_module, "timed_command", fake_timed_command)
    checkpoint = _checkpoint(tmp_path / "model.alignair")
    runner = make_runner("alignair-api@cand", RunnerOptions(checkpoints={"cand": str(checkpoint)}))
    runner._setup = {"model_reference": {"path": str(tmp_path / "model_reference.fasta")}}
    runner._reference = reference
    fasta = tmp_path / "reads.fasta"
    fasta.write_text(">r1\nACGT\n")
    result = runner.run(fasta, tmp_path / "out.tsv", suite="W-97", arm="shared", log_dir=tmp_path / "logs")
    assert result.tool == "alignair-api@cand" and result.returncode == 0
    # a failure before the tool runs is recorded under the full id too
    failed = runner.run(fasta, tmp_path / "out.tsv", suite="W-97", arm="genotype", log_dir=tmp_path / "logs2",
                        alleles=["IGHVX9*01"])
    assert failed.tool == "alignair-api@cand" and failed.returncode == 1


def test_a_variants_paths_are_single_components(tmp_path):
    layout = RunLayout(Path("/run"))
    tool = "alignair-api@cand"
    assert layout.run_dir(tool, "S1", "shared") == Path("/run/runs/alignair-api@cand/S1/shared")
    assert layout.work_dir(tool) == Path("/run/runs/alignair-api@cand/_work")
    assert layout.tool_scores("S1", "shared", tool) == Path("/run/scores/S1/shared/alignair-api@cand.scores.jsonl.gz")
    assert layout.real_run_json(tool, "W-97", "genotype") == Path(
        "/run/real/W-97/runs/alignair-api@cand/genotype/run.json")
    pair = "alignair-api@a_vs_alignair-api@b"
    assert layout.real_comparison_json("W-97", "shared", pair) == Path(f"/run/real/W-97/scores/shared/{pair}.json")
    # every tool id parse_tool accepts is one path component
    for label in ("a", "..x", "x..", "-a", "a.b", "_"):
        assert layout.run_dir(f"alignair-local@{label}", "S1", "shared").parent.parent.name == f"alignair-local@{label}"
    with pytest.raises(ValueError, match="single path component"):
        layout.run_dir("alignair-api@a/b", "S1", "shared")


# --- the plain tools, as before ----------------------------------------------------------------------------------

def _plain(found: dict, checkpoint: Path) -> dict:
    assert found["checkpoint"] == str(checkpoint)
    return {**found, "checkpoint": RECORD["placeholder"]}


def test_the_plain_tools_records_are_unchanged(tmp_path, fixed_code):
    checkpoint = _checkpoint(tmp_path / "model.alignair", RECORD["checkpoint_bytes"].encode())
    # a variant's checkpoints beside it change nothing of a plain tool
    options = RunnerOptions(alignair_checkpoint=str(checkpoint), device="cpu",
                            checkpoints={"a": str(_checkpoint(tmp_path / "a.alignair", b"a"))})
    fingerprints = {}
    for tool in ("alignair-local", "alignair-api"):
        runner = make_runner(tool, options)
        assert runner.name == tool and not hasattr(runner, "label")
        fingerprints[tool] = _plain(runner.fingerprint(), checkpoint)
    assert fingerprints == RECORD["fingerprints"]
    # the all identity and its hash
    settings = {"oracle": {}, "igblast": {"igblastn_sha256": "i" * 64},
                "alignair-local": dict(fingerprints["alignair-local"])}
    cfg = RunConfig(suites=("S1",), tools=("igblast", "alignair-local"), alignair_checkpoint=str(checkpoint),
                    checkpoints={"a": str(tmp_path / "a.alignair")})
    found = identity(cfg, settings)
    assert found == RECORD["all_identity"] and digest(found) == RECORD["all_config_hash"]
    assert sorted(found) == RECORD["all_manifest_identity_keys"]
    # the real manifest's identity and each plain tool's settings in it
    igblast = tmp_path / "igblast"
    for name in ("bin/igblastn", "bin/makeblastdb", "optional_file/human_gl.aux"):
        _checkpoint(igblast / name, name.encode())
    image = make_runner("alignair", RunnerOptions(cache_dir=str(tmp_path / "cache")))
    image._inspect_image = lambda: {"id": "sha256:" + "i" * 64}
    image.install_model = lambda: {"model": {"sha256": "m" * 64}}
    real_settings = {"igblast": make_runner("igblast", RunnerOptions(igblast_dir=str(igblast))).fingerprint(),
                     "alignair": image.fingerprint(), "alignair-api": fingerprints["alignair-api"]}
    assert {tool: sorted(values) for tool, values in real_settings.items()} == RECORD["real_manifest_settings_keys"]
    real_cfg = RealConfig(donors=("W-97",), split="dev", data_root=str(tmp_path),
                          alignair_checkpoint=str(checkpoint), checkpoints={"a": str(tmp_path / "a.alignair")})
    reference = SimpleNamespace(digest=lambda: "r" * 64, gapped_digest=lambda: "g" * 64)
    real_identity = manifest_identity(real_cfg, reference, real_settings)
    assert sorted(real_identity) == RECORD["real_manifest_identity_keys"]
    assert real_identity["settings"] == real_settings
    # the paths
    layout = RunLayout(Path("/run"))
    for tool in ("alignair-local", "alignair-api"):
        paths = [str(p) for p in (
            layout.run_dir(tool, "S1", "shared"), layout.predictions(tool, "S1", "shared"),
            layout.run_json(tool, "S1", "shared"), layout.logs_dir(tool, "S1", "shared"),
            layout.run_dir(tool, "S5", "genotype"), layout.donor_predictions(tool, "S5", "genotype", "d1"),
            layout.work_dir(tool), layout.tool_scores("S1", "shared", tool),
            layout.tool_report_json("S1", "shared", tool), layout.real_run_dir(tool, "W-97", "shared"),
            layout.real_predictions(tool, "W-97", "genotype"), layout.real_run_json(tool, "W-97", "shared"),
            layout.real_scores("W-97", "shared", tool),
            layout.real_comparison_json("W-97", "shared", f"igblast_vs_{tool}"),
            layout.real_comparison_json("W-97", "arms", tool))]
        assert paths == RECORD["paths"][tool]
    # the real run identity: the same fields, and the same runner modules behind its code digest
    reads = tmp_path / "reads.fasta"
    reads.write_text(">r1\nACGT\n")
    real = real_steps.run_identity("alignair-api", reads, RECORD["fingerprints"]["alignair-api"],
                                   reference_digest="r" * 64)
    code = real.pop("code_sha256")
    assert real == RECORD["real_run_identity_without_code"]
    assert {tool: list(files) for tool, files in real_steps.RUNNER_CODE.items()} == RECORD["runner_code"]
    assert code == real_steps.code_digest(RECORD["runner_code"]["alignair-api"])


def test_a_variants_run_code_adds_the_variant_module():
    assert real_steps.runner_code("alignair-api") == real_steps.RUNNER_CODE["alignair-api"]
    assert real_steps.runner_code("igblast") == real_steps.RUNNER_CODE["igblast"]
    assert real_steps.runner_code("alignair-api@b") == (*real_steps.RUNNER_CODE["alignair-api"],
                                                        "runners/variants.py")


# --- the command line's --checkpoint LABEL=PATH ------------------------------------------------------------------

def test_checkpoint_option_parses_label_and_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert checkpoint_spec("cand=models/c.alignair") == ("cand", str(tmp_path / "models" / "c.alignair"))
    assert checkpoint_spec("a=~/m=1.alignair") == ("a", str(Path.home() / "m=1.alignair"))
    for bad in ("cand", "=x.alignair", "cand=", "a/b=x.alignair", "a b=x"):
        with pytest.raises(argparse.ArgumentTypeError):
            checkpoint_spec(bad)
    assert checkpoint_map([("a", "/m/a"), ("b", "/m/b")]) == {"a": "/m/a", "b": "/m/b"}
    assert checkpoint_map([]) == {} and checkpoint_map(None) == {}
    with pytest.raises(ValueError, match="label 'a' is given twice"):
        checkpoint_map([("a", "/m/a"), ("a", "/m/b")])


def test_run_takes_plain_tools_only(tmp_path, capsys):
    # run has no --checkpoint LABEL=PATH, so a variant is refused with where to run one instead
    checkpoint = _checkpoint(tmp_path / "model.alignair")
    for tool in ("alignair-api@x", "alignair-local@x"):
        assert main(["run", "--run-dir", str(tmp_path / "run"), "--tool", tool, "--suites", "S0", "--checkpoint",
                     str(checkpoint)]) == 2
        err = capsys.readouterr().err
        assert f"tool {tool!r}: run takes plain tools; use all or real for checkpoint variants" in err, err
        assert "x=PATH" not in err
    assert not (tmp_path / "run").exists()
