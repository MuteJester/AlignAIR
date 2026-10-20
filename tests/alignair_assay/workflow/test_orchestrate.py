"""The ``all`` orchestration: manifest and stale-config refusal, resumable build and runs, scoring, the run summary
and the run report, on the oracle and fake tools registered for the test."""
import dataclasses
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("GenAIRR")

from alignair_assay.cli import main  # noqa: E402
from alignair_assay.core.layout import RunLayout  # noqa: E402
from alignair_assay.core.provenance import sha256_file  # noqa: E402
from alignair_assay.runners import RUNNERS  # noqa: E402
from alignair_assay.runners.base import RunResult, fasta_ids, read_run_json  # noqa: E402
from alignair_assay.runners.oracle import OracleRunner  # noqa: E402
from alignair_assay.suites import SUITES, get_suite  # noqa: E402
from alignair_assay.workflow.orchestrate import RunConfig, config_hash, run_all  # noqa: E402

JUNCTION_COLUMNS = ("junction", "junction_aa", "junction_start", "junction_end")


class HalfOracle(OracleRunner):
    """Oracle rows with the junction blanked on every other read (the second, fourth, ... in FASTA order)."""

    name = "half"

    def run(self, fasta, out_tsv, **kwargs):
        result = super().run(fasta, out_tsv, **kwargs)
        header, *rows = [line.split("\t") for line in Path(out_tsv).read_text().splitlines()]
        blank = [header.index(column) for column in JUNCTION_COLUMNS]
        for row in rows[1::2]:
            for index in blank:
                row[index] = ""
        Path(out_tsv).write_text("".join("\t".join(row) + "\n" for row in [header, *rows]))
        return dataclasses.replace(result, tool=self.name)


class BlankingOracle(HalfOracle):
    """A broken oracle: the half-blanked rows under the oracle's name."""

    name = "oracle"


class GenotypeCrash(OracleRunner):
    """Oracle rows on the shared arm; on a restricted (genotype) run it fails as AlignAIR 3.0.0 does."""

    name = "picky"

    def run(self, fasta, out_tsv, *, suite, arm, log_dir, alleles=None):
        if alleles is None:
            result = super().run(fasta, out_tsv, suite=suite, arm=arm, log_dir=log_dir)
            return dataclasses.replace(result, tool=self.name)
        return RunResult(tool=self.name, suite=suite, arm=arm, predictions=str(out_tsv), n_reads=len(fasta_ids(fasta)),
                         wall_seconds=0.1, peak_rss_kb=None, returncode=1, commands=[["picky", str(fasta)]],
                         provenance={"error": "AttributeError: 'tuple' object has no attribute 'items'"})


@pytest.fixture(autouse=True)
def fake_tools(monkeypatch):
    monkeypatch.setitem(RUNNERS, "half", lambda options: HalfOracle())
    monkeypatch.setitem(RUNNERS, "picky", lambda options: GenotypeCrash())


def _all(root, **overrides) -> tuple[RunConfig, int]:
    options = dict(suites=("S0", "S1"), tools=("oracle", "half"), scale=0.02, workers=2) | overrides
    force = options.pop("force", False)
    cfg = RunConfig(**options)
    return cfg, run_all(cfg, RunLayout(root), force=force)


@pytest.fixture(scope="module")
def half_run(tmp_path_factory):
    """``all`` on S0 and S1 at scale 0.02 with the oracle against ``half``, run once for the module."""
    root = tmp_path_factory.mktemp("half")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(RUNNERS, "half", lambda options: HalfOracle())
        cfg, status = _all(root)
    return root, cfg, status


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory):
    """``all`` on S0 alone at scale 0.2 (10 cases), run once for the module."""
    root = tmp_path_factory.mktemp("toy")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(RUNNERS, "half", lambda options: HalfOracle())
        cfg, status = _all(root, suites=("S0",), scale=0.2, workers=1)
    assert status == 0
    return root, cfg


def _copy(root: Path, tmp_path: Path) -> Path:
    target = tmp_path / "run"
    shutil.copytree(root, target)
    return target


def _report(root: Path) -> dict:
    return json.loads(RunLayout(root).report_json.read_text())


def _suite(report: dict, name: str) -> dict:
    return next(suite for suite in report["suites"] if suite["name"] == name)


def _outputs(layout: RunLayout, cfg: RunConfig) -> dict[Path, int]:
    """Modification times of every build and run output."""
    paths = [layout.manifest]
    for suite in cfg.suites:
        paths += [layout.cases(suite), layout.truth(suite), layout.reads(suite), layout.build_json(suite)]
        for tool in ("oracle", *cfg.tools):
            paths += [layout.predictions(tool, suite, "shared"), layout.run_json(tool, suite, "shared")]
    return {path: path.stat().st_mtime_ns for path in dict.fromkeys(paths)}


def test_all_end_to_end_with_oracle_and_fake(half_run, tmp_path):
    root, cfg, status = half_run
    layout = RunLayout(root)
    assert status == 0

    manifest = json.loads(layout.manifest.read_text())
    assert manifest["config_hash"] == config_hash(cfg) and manifest["split"] == "dev"
    assert manifest["config"]["suites"] == ["S0", "S1"] and manifest["config"]["scale"] == 0.02
    assert {"host", "git", "genairr"} <= set(manifest)
    assert json.loads(layout.reference_json.read_text())["source"] == "truth"

    report = _report(root)
    assert report["kind"] == "alignair_assay.run_report" and (report["a"], report["b"]) == ("oracle", "half")
    assert report["failed_runs"] == [] and report["oracle_failures"] == []
    s1 = _suite(report, "S1")
    assert (s1["title"], s1["cases"], s1["excluded"]) == ("Core repertoire", 100, 0)
    assert s1["oracle"]["status"] == "perfect" and s1["oracle"]["cases"] == 100
    shared = s1["arms"]["shared"]
    assert shared["compared"] and shared["link"] == "scores/S1/shared/comparison.md"
    assert list(shared["rows"]) == [m for m in get_suite("S1").primary if m in shared["rows"]]
    assert shared["rows"]["junction.nt_exact"]["verdict"] == "a_better"
    assert shared["rows"]["v.top1_in_set"]["verdict"] == "tie_or_inconclusive"
    assert shared["verdicts"]["a_better"] >= 1 and sum(shared["verdicts"].values()) == len(shared["rows"])
    assert _suite(report, "S0")["oracle"]["status"] == "perfect"

    # Scores and the comparison are the same artifacts the score and compare commands write.
    scores = layout.scores_dir("S1", "shared")
    for tool in ("oracle", "half"):
        for path in (layout.tool_scores("S1", "shared", tool), layout.tool_report_json("S1", "shared", tool),
                     layout.tool_report_md("S1", "shared", tool)):
            assert path.is_file() and path.parent == scores
    comparison = json.loads(layout.comparison_json("S1", "shared").read_text())
    assert main(["compare", "--truth", str(layout.truth("S1")),
                 "--a", str(layout.predictions("oracle", "S1", "shared")), "--a-name", "oracle",
                 "--b", str(layout.predictions("half", "S1", "shared")), "--b-name", "half",
                 "--out-dir", str(tmp_path / "cmp")]) == 0
    by_command = json.loads((tmp_path / "cmp" / "comparison.json").read_text())
    assert by_command["comparison"] == comparison["comparison"]
    assert by_command["by_slice"] == comparison["by_slice"]
    assert comparison["provenance"]["command"] == "all"
    # A suite's comparison markdown slices its own primary metrics (S0: V, J and the junction only).
    s0_slices = layout.comparison_md("S0", "shared").read_text().split("## By stratum")[1]
    assert "| v.top1_in_set |" in s0_slices and "| clone_key.exact |" not in s0_slices

    md = layout.report_md.read_text()
    assert md.startswith("# AlignAIR assay run: oracle vs half")
    assert "## Failed runs" not in md and "## Oracle check failures" not in md
    assert md.index("## Summary") < md.index("## Suites")
    assert "(scores/S1/shared/comparison.md)" in md and "### S1: Core repertoire" in md


def test_resume_skips_completed_steps(half_run, tmp_path, capsys):
    root = _copy(half_run[0], tmp_path)
    cfg = half_run[1]
    layout = RunLayout(root)
    before = _outputs(layout, cfg)

    _, status = _all(root)
    assert status == 0
    assert _outputs(layout, cfg) == before
    printed = capsys.readouterr().out
    assert "build S1: up to date, skipped" in printed
    assert "run half S1/shared: up to date, skipped" in printed
    assert "run oracle S0/shared: up to date, skipped" in printed
    assert _report(root)["failed_runs"] == []

    # A changed input is redone: a run whose reads changed is re-run, a failed run is retried.
    record = read_run_json(layout.run_json("half", "S0", "shared"))
    stale = dataclasses.replace(record, provenance={**record.provenance, "reads": {"path": "x", "sha256": "0"}})
    layout.run_json("half", "S0", "shared").write_text(json.dumps(dataclasses.asdict(stale)))
    _, status = _all(root)
    assert status == 0
    assert "run half S0/shared: return code 0" in capsys.readouterr().out
    assert read_run_json(layout.run_json("half", "S0", "shared")).provenance["reads"]["sha256"] == sha256_file(
        layout.reads("S0"))


def test_stale_config_refused(toy_run, tmp_path):
    root = _copy(toy_run[0], tmp_path)
    cfg = toy_run[1]
    layout = RunLayout(root)
    before = _outputs(layout, cfg)
    with pytest.raises(ValueError, match=r"seed 20261019 -> 7") as error:
        _all(root, suites=("S0",), scale=0.2, workers=1, seed=7)
    assert "--force" in str(error.value) and "scale" not in str(error.value)
    assert _outputs(layout, cfg) == before  # nothing touched

    # Execution settings are not part of the configuration.
    assert _all(root, suites=("S0",), scale=0.2, workers=2, threads=3)[1] == 0

    # --force proceeds: the suite is rebuilt with the new seed and every run is redone on the new reads.
    _, status = _all(root, suites=("S0",), scale=0.2, workers=1, seed=7, force=True)
    assert status == 0
    manifest = json.loads(layout.manifest.read_text())
    assert manifest["config"]["seed"] == 7 and manifest["replaced"]["changed"] == ["seed"]
    assert json.loads(layout.build_json("S0").read_text())["seed"] == 7
    for tool in ("oracle", "half"):
        record = read_run_json(layout.run_json(tool, "S0", "shared"))
        assert record.provenance["reads"]["sha256"] == sha256_file(layout.reads("S0"))
    assert _report(root)["config"]["seed"] == 7


def test_split_change_refused(toy_run, tmp_path):
    root = _copy(toy_run[0], tmp_path)
    with pytest.raises(ValueError, match=r"split dev -> test"):
        _all(root, suites=("S0",), scale=0.2, workers=1, split="test")
    with pytest.raises(ValueError, match=r"suites \['S0'\] -> \['S0', 'S11'\]"):
        _all(root, suites=("S0", "S11"), scale=0.2, workers=1)
    with pytest.raises(ValueError, match="tools"):
        _all(root, suites=("S0",), scale=0.2, workers=1, tools=("half", "oracle"))


def test_config_checked_before_anything_runs(tmp_path):
    for overrides, message in ((dict(tools=("oracle",)), "two tools"), (dict(tools=("half", "half")), "two tools"),
                               (dict(tools=("oracle", "nope")), "unknown tool 'nope'"),
                               (dict(suites=("S99",)), "unknown suite"), (dict(scale=0), "scale"),
                               (dict(split="other"), "unknown split"), (dict(speed=True), "speed")):
        with pytest.raises(ValueError, match=message):
            _all(tmp_path / "run", **overrides)
        assert not (tmp_path / "run").exists()


def test_failed_run_returns_2_and_is_reported(tmp_path, monkeypatch):
    cohort = get_suite("S0_COHORT")
    monkeypatch.setitem(SUITES, "S0_COHORT", dataclasses.replace(cohort, arms=("shared", "genotype")))
    root = tmp_path / "run"
    _, status = _all(root, suites=("S0_COHORT",), tools=("oracle", "picky"), scale=1.0, workers=1)
    assert status == 2
    layout = RunLayout(root)

    report = _report(root)
    assert report["failed_runs"] == [{
        "tool": "picky", "suite": "S0_COHORT", "arm": "genotype", "status": "failed", "returncode": 1,
        "failed_donor": "donor00", "error": "AttributeError: 'tuple' object has no attribute 'items'",
        "run_json": "runs/picky/S0_COHORT/genotype/run.json"}]
    suite = _suite(report, "S0_COHORT")
    assert suite["oracle"]["status"] == "perfect"
    assert suite["arms"]["shared"]["compared"] and not suite["arms"]["genotype"]["compared"]
    assert not layout.comparison_json("S0_COHORT", "genotype").exists()
    assert layout.tool_scores("S0_COHORT", "genotype", "oracle").is_file()  # the oracle's genotype arm is scored
    assert not layout.tool_scores("S0_COHORT", "genotype", "picky").exists()

    (genotype,) = report["genotype_arms"]
    row = next(row for row in genotype["rows"] if row["metric"] == "v.top1_in_set")
    assert row["values"] == {"oracle": {"shared": 1.0, "genotype": 1.0}, "picky": {"shared": 1.0, "genotype": None}}

    md = layout.report_md.read_text()
    section = md[md.index("## Failed runs"):md.index("## Summary")]
    assert "| picky | S0_COHORT | genotype | failed | 1 | donor00 |" in section
    assert "AttributeError: 'tuple' object has no attribute 'items'" in section
    assert "## Genotype arms" in md

    # The failed run is retried on resume (and fails again); the completed ones are not.
    assert _all(root, suites=("S0_COHORT",), tools=("oracle", "picky"), scale=1.0, workers=1)[1] == 2


def test_oracle_failure_flagged(toy_run, tmp_path, monkeypatch):
    root = _copy(toy_run[0], tmp_path)
    layout = RunLayout(root)
    shutil.rmtree(layout.run_dir("oracle", "S0", "shared"))
    monkeypatch.setitem(RUNNERS, "oracle", lambda options: BlankingOracle())
    _, status = _all(root, suites=("S0",), scale=0.2, workers=1)
    assert status == 2

    report = _report(root)
    assert report["failed_runs"] == []
    (failure,) = report["oracle_failures"]
    assert failure["suite"] == "S0" and failure["status"] == "imperfect" and failure["problems"] > 0
    case_id, metric, _ = _suite(report, "S0")["oracle"]["examples"][0]
    assert case_id.startswith("S0:") and (metric.startswith("junction.") or metric == "clone_key.exact")
    md = layout.report_md.read_text()
    assert "## Oracle check failures" in md and "| S0 | imperfect |" in md


def test_s11_parity_section(tmp_path):
    root = tmp_path / "run"
    _, status = _all(root, suites=("S11",), workers=2)
    assert status == 0
    parity = _report(root)["s11_parity"]
    assert parity["alpha"] == 0.05
    oracle, half = parity["tools"]["oracle"], parity["tools"]["half"]
    assert oracle["pairs"] == half["pairs"] == 20
    assert oracle["differ"] == []
    # half blanks every reverse-complemented read's junction: the forward reads are better.
    assert "junction.nt_exact" in half["differ"]
    row = next(row for row in half["rows"] if row["metric"] == "junction.nt_exact")
    assert (row["forward"], row["reverse"], row["n_paired"], row["verdict"]) == (1.0, 0.0, 20, "a_better")
    assert [row["metric"] for row in half["rows"]][0] == "orientation.correct"
    md = RunLayout(root).report_md.read_text()
    section = md[md.index("## S11 orientation parity"):]
    half_line = next(line for line in section.splitlines() if line.startswith("**half**"))
    assert half_line.startswith("**half** (20 pairs): differs on ") and "junction.nt_exact" in half_line
    assert "**oracle** (20 pairs): no metric differs" in section


def test_report_command_rescores(half_run, tmp_path, capsys):
    root = _copy(half_run[0], tmp_path)
    layout = RunLayout(root)
    layout.report_md.unlink()
    assert main(["report", "--run-dir", str(root)]) == 0
    assert layout.report_md.is_file() and _report(root)["a"] == "oracle"
    assert "report.md" in capsys.readouterr().out
    # Swapped tools: the comparison is redone in the new direction.
    assert main(["report", "--run-dir", str(root), "--a", "half", "--b", "oracle"]) == 0
    comparison = json.loads(layout.comparison_json("S1", "shared").read_text())
    assert (comparison["a"], comparison["b"]) == ("half", "oracle")
    assert comparison["comparison"]["junction.nt_exact"]["verdict"] == "b_better"
    assert main(["report", "--run-dir", str(tmp_path / "empty")]) == 2
    assert "manifest.json" in capsys.readouterr().err
