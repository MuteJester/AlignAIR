"""``all`` with checkpoint variants: two labels of one base are the two compared tools.

The variants are alignair-local's (``runners.variants``) with their real fingerprints (the checkpoint's digest and
the label), whose runs write oracle rows instead of running the CLI, so the same checkpoint under two labels must
compare to zero on every metric. A variant's checkpoint changed after its run is another configuration (refused,
then redone alone when forced) and a stale run in the report; a missing or unknown one is refused before anything
is written.
"""
import dataclasses
import json
from pathlib import Path

import pytest

pytest.importorskip("GenAIRR")

from alignair_assay.cli import main  # noqa: E402
from alignair_assay.cli import all as all_command  # noqa: E402
from alignair_assay.core.layout import RunLayout  # noqa: E402
from alignair_assay.core.provenance import sha256_file  # noqa: E402
from alignair_assay.runners import VARIANTS  # noqa: E402
from alignair_assay.runners import alignair as alignair_module  # noqa: E402
from alignair_assay.runners.base import read_run_json  # noqa: E402
from alignair_assay.runners.oracle import OracleRunner  # noqa: E402
from alignair_assay.runners.variants import AlignAIRLocalVariant  # noqa: E402
from alignair_assay.workflow.orchestrate import RunConfig, check_config, run_all, score_and_report  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.genairr_build]  # these tests build GenAIRR suites
CODE = {"code_sha256": "c" * 64, "torch_version": None}
TOOLS = ("alignair-local@a", "alignair-local@b")


class OracleVariant(AlignAIRLocalVariant):
    """alignair-local@<label> with its real fingerprint, whose runs write oracle rows and record only their reads."""

    def prepare(self, layout, reference):
        self._oracle = OracleRunner(layout)
        return {}

    def run(self, fasta, out_tsv, **kwargs):
        result = self._oracle.run(fasta, out_tsv, **kwargs)
        return dataclasses.replace(result, tool=self.name, provenance={})


def _patch(patch) -> None:
    patch.setitem(VARIANTS, "alignair-local", OracleVariant.from_label)
    patch.setattr(alignair_module, "local_code_identity", lambda: dict(CODE))


@pytest.fixture(autouse=True)
def oracle_variants(monkeypatch):
    _patch(monkeypatch)


def _config(checkpoints: dict, **overrides) -> RunConfig:
    return RunConfig(**(dict(suites=("S0",), tools=TOOLS, scale=0.2, workers=1, checkpoints=checkpoints)
                        | overrides))


@pytest.fixture(scope="module")
def null_run(tmp_path_factory):
    """``all`` on S0 (10 cases) with one checkpoint under labels a and b, run once for the module."""
    root = tmp_path_factory.mktemp("variants")
    checkpoint = root.parent / f"{root.name}.alignair"
    checkpoint.write_bytes(b"weights")
    checkpoints = {"a": str(checkpoint), "b": str(checkpoint)}
    with pytest.MonkeyPatch.context() as patch:
        _patch(patch)
        status = run_all(_config(checkpoints), RunLayout(root / "run"))
    return root / "run", checkpoints, status


def _copy(root: Path, tmp_path: Path) -> Path:
    import shutil

    target = tmp_path / "run"
    shutil.copytree(root, target)
    return target


def test_one_checkpoint_under_two_labels_compares_to_zero(null_run):
    root, checkpoints, status = null_run
    layout = RunLayout(root)
    assert status == 0
    manifest = json.loads(layout.manifest.read_text())
    assert manifest["identity"]["tools"] == list(TOOLS)
    assert manifest["config"]["checkpoints"] == checkpoints
    digest = sha256_file(checkpoints["a"])
    for tool, label in zip(TOOLS, "ab"):
        settings = manifest["identity"]["settings"][tool]
        assert (settings["label"], settings["checkpoint"], settings["checkpoint_sha256"]) == (
            label, checkpoints[label], digest)
        record = read_run_json(layout.run_json(tool, "S0", "shared"))
        assert record.tool == tool and record.provenance["settings"] == settings
        assert layout.run_json(tool, "S0", "shared").parent == root / "runs" / tool / "S0" / "shared"
    comparison = json.loads(layout.comparison_json("S0", "shared").read_text())
    assert (comparison["a"], comparison["b"]) == TOOLS
    rows = comparison["comparison"]
    assert rows and all(row["delta"] in (0, None) for row in rows.values())
    assert all(row["wins"] == row["losses"] == 0 for row in rows.values())
    assert {row["verdict"] for row in rows.values()} == {"tie_or_inconclusive"}
    # two checkpoints: the reads the second fixes and breaks are counted (none), per stratum too
    assert comparison["changes"]["reads"] == {"fixed": 0, "broken": 0, "fixed_examples": [], "broken_examples": [],
                                              "groups": {}}
    assert all(found["fixed"] == found["broken"] == 0 for found in comparison["changes"]["metrics"].values())
    assert ("alignair-local@b fixes 0 reads alignair-local@a gets wrong and breaks 0 it gets right"
            in layout.comparison_md("S0", "shared").read_text())
    report = json.loads(layout.report_json.read_text())
    assert (report["a"], report["b"]) == TOOLS
    assert report["checkpoints"] == {tool: {"label": label, "path": checkpoints[label], "sha256": digest,
                                            "code_sha256": CODE["code_sha256"]}
                                     for tool, label in zip(TOOLS, "ab")}
    md = layout.report_md.read_text()
    assert md.startswith("# AlignAIR assay run: alignair-local@a vs alignair-local@b")
    assert report["suites"][0]["arms"]["shared"]["changes"]["fixed"] == 0
    assert "Reads alignair-local@b fixes / breaks against alignair-local@a: 0 / 0." in md
    # both checkpoints ran under the AlignAIR code of this checkout: the report says which
    assert (f"Checkpoints: alignair-local@a `{digest[:12]}` ({checkpoints['a']}), alignair-local@b "
            f"`{digest[:12]}` ({checkpoints['b']}); both ran under AlignAIR code `{'c' * 12}`.") in md


def test_a_baseline_given_second_is_warned_of(tmp_path, capsys):
    checkpoint = tmp_path / "model.alignair"
    checkpoint.write_bytes(b"weights")
    cfg = _config({"base": str(checkpoint), "cand": str(checkpoint)},
                  tools=("alignair-local@cand", "alignair-local@base"))
    assert run_all(cfg, RunLayout(tmp_path / "run")) == 0
    assert ("warning: alignair-local@base comes after alignair-local@cand, so Δ = alignair-local@base − "
            "alignair-local@cand: give the baseline first") in capsys.readouterr().err


def test_a_checkpoint_retrained_in_place_makes_a_plain_tools_runs_stale_too(tmp_path):
    from alignair_assay.workflow.summary import checkpoint_changed

    path = tmp_path / "model.alignair"
    path.write_bytes(b"weights")
    plain = {"checkpoint": str(path), "checkpoint_sha256": sha256_file(path), "device": "cpu", **CODE}
    variant = {**plain, "label": "a"}
    assert checkpoint_changed(plain) is None and checkpoint_changed(variant) is None
    path.write_bytes(b"retrained")
    for settings in (plain, variant):  # alignair-local as well as alignair-local@a
        assert checkpoint_changed(settings) == f"its checkpoint {path} changed since this run (another SHA-256)"
    path.unlink()
    assert checkpoint_changed(plain) == f"its checkpoint {path} is missing since this run"
    # a tool that runs no checkpoint file (the alignair image, igblast) has nothing to check
    assert checkpoint_changed({"image": "alignair", "model": "alignair-igh-human@1.0.0"}) is None
    assert checkpoint_changed(None) is None


def test_a_changed_checkpoint_is_another_configuration_and_a_stale_run(null_run, tmp_path, capsys):
    root = _copy(null_run[0], tmp_path)
    layout = RunLayout(root)
    other = tmp_path / "other.alignair"
    other.write_bytes(b"weights")
    checkpoints = {**null_run[1], "b": str(other)}
    # the same bytes elsewhere: the path is part of b's fingerprint, so b alone is redone once forced
    with pytest.raises(ValueError, match=r"alignair-local@b\.checkpoint "):
        run_all(_config(checkpoints), layout)
    capsys.readouterr()
    assert run_all(_config(checkpoints), layout, force=True) == 0
    printed = capsys.readouterr().out
    assert "run alignair-local@a S0/shared: up to date, skipped" in printed
    assert "run alignair-local@b S0/shared: return code 0" in printed
    # retrained in place: its runs no longer match their record, so the report lists them as stale
    other.write_bytes(b"retrained")
    assert score_and_report(_config(checkpoints), layout, command="report") == 2
    failed = json.loads(layout.report_json.read_text())["failed_runs"]
    assert {(run["tool"], run["status"]) for run in failed} == {("alignair-local@b", "stale")}
    assert "checkpoint" in failed[0]["error"]
    # and all refuses the configuration, naming the variant's digest, until forced; then b alone is redone
    with pytest.raises(ValueError, match=r"alignair-local@b\.checkpoint_sha256"):
        run_all(_config(checkpoints), layout)
    capsys.readouterr()
    assert run_all(_config(checkpoints), layout, force=True) == 0
    printed = capsys.readouterr().out
    assert "run alignair-local@a S0/shared: up to date, skipped" in printed
    assert "run alignair-local@b S0/shared: return code 0" in printed
    # a checkpoint gone after its run is stale too
    other.unlink()
    assert score_and_report(_config(checkpoints), layout, command="report") == 2
    failed = json.loads(layout.report_json.read_text())["failed_runs"]
    assert {(run["tool"], run["status"]) for run in failed} == {("alignair-local@b", "stale")}


def test_variants_are_refused_before_anything_is_written(tmp_path):
    checkpoint = tmp_path / "model.alignair"
    checkpoint.write_bytes(b"weights")
    good = {"a": str(checkpoint), "b": str(checkpoint)}
    for overrides, error, message in (
            (dict(checkpoints={"a": str(checkpoint)}), ValueError, r"unknown checkpoint label 'b'.*known labels: a"),
            (dict(checkpoints={**good, "b": str(tmp_path / "gone.alignair")}), FileNotFoundError,
             r"alignair-local@b: no AlignAIR checkpoint"),
            (dict(tools=("alignair-local@a", "alignair-local@a")), ValueError, "two tools"),
            (dict(tools=("alignair-local@a", "alignair-local@a/b")), ValueError, "label"),
            (dict(tools=("alignair-local@a", "igblast@b")), ValueError, "take a checkpoint label"),
            (dict(speed=True, suites=("S1",)), ValueError, "does not time checkpoint variants"),
            # a checkpoint given under a label no tool runs is a mistake, never ignored
            (dict(checkpoints={**good, "c": str(checkpoint), "cand": str(checkpoint)}), ValueError,
             r"no tool runs --checkpoint label 'c', 'cand'")):
        cfg = _config(overrides.pop("checkpoints", good), **overrides)
        with pytest.raises(error, match=message):
            check_config(cfg)
        with pytest.raises(error, match=message):
            run_all(cfg, RunLayout(tmp_path / "run"))
        assert not (tmp_path / "run").exists()


def test_the_cli_takes_repeated_checkpoints(tmp_path, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(all_command, "run_all", lambda cfg, layout, force=False: seen.append(cfg) or 0)
    args = ["all", "--run-dir", str(tmp_path / "run"), "--suites", "S0", "--tools", ",".join(TOOLS),
            "--checkpoint", "a=~/a.alignair", "--checkpoint", f"b={tmp_path / 'b.alignair'}"]
    assert main(args) == 0
    assert seen[0].checkpoints == {"a": str(Path.home() / "a.alignair"), "b": str(tmp_path / "b.alignair")}
    assert seen[0].tools == TOOLS
    assert main([*args, "--checkpoint", "a=/elsewhere.alignair"]) == 2
    assert "label 'a' is given twice" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main([*args, "--checkpoint", "a/b=x.alignair"])
    assert "checkpoint label 'a/b'" in capsys.readouterr().err
