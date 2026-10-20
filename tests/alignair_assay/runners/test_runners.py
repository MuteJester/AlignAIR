"""Runners: the shared reference, timed commands, run records, the oracle and IgBLAST runners and the run command.

The IgBLAST tests run only when ``ALIGNAIR_ASSAY_IGBLAST_DIR`` names an unpacked IgBLAST release.
"""
import dataclasses
import json
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("GenAIRR")

from alignair_assay.airr.reader import load_airr_tsv  # noqa: E402
from alignair_assay.cli import main  # noqa: E402
from alignair_assay.core.jsonl import read_header, read_jsonl, write_jsonl  # noqa: E402
from alignair_assay.core.layout import RunLayout  # noqa: E402
from alignair_assay.core.provenance import sha256_file  # noqa: E402
from alignair_assay.core.reference import Reference  # noqa: E402
from alignair_assay.runners import RUNNERS, make_runner  # noqa: E402
from alignair_assay.runners.base import (  # noqa: E402
    RunnerOptions,
    RunResult,
    read_run_json,
    require_successful_run,
    timed_command,
    write_run_json,
)
from alignair_assay.runners.igblast import IgBlastRunner  # noqa: E402
from alignair_assay.runners.oracle import OracleRunner  # noqa: E402
from alignair_assay.runners.reference import (  # noqa: E402
    ensure_shared_reference,
    verify_reference,
    write_reference_fasta,
    write_segment_fastas,
)
from alignair_assay.scoring.pipeline import score_tool  # noqa: E402
from alignair_assay.testing import assert_oracle_perfect  # noqa: E402
from alignair_assay.truth.io import read_truth_jsonl  # noqa: E402
from alignair_assay.workflow.build import build_suite  # noqa: E402

IGBLAST_DIR = os.environ.get("ALIGNAIR_ASSAY_IGBLAST_DIR")
needs_igblast = pytest.mark.skipif(not IGBLAST_DIR,
                                   reason="set ALIGNAIR_ASSAY_IGBLAST_DIR to an unpacked IgBLAST release")


@pytest.fixture(scope="module")
def built_run(tmp_path_factory, reference):
    """S0 at scale 1.0 (50 cases), built once for the module."""
    root = tmp_path_factory.mktemp("built")
    build_suite("S0", RunLayout(root), reference_name="HUMAN_IGH_OGRDB", split="dev", seed=5, scale=1.0, workers=1)
    return root


@pytest.fixture
def run_layout(built_run, tmp_path):
    """A private copy of the built run directory."""
    root = tmp_path / "run"
    shutil.copytree(built_run, root)
    return RunLayout(root)


def _without(reference: Reference, *names: str) -> Reference:
    return Reference(alleles={g: {n: s for n, s in table.items() if n not in names}
                              for g, table in reference.alleles.items()}, anchors={})


def _run(layout: RunLayout, tool: str, *extra: str) -> int:
    return main(["run", "--run-dir", str(layout.root), "--tool", tool, "--suites", "S0", *extra])


# --- shared reference -------------------------------------------------------------------------------------------

def test_reference_from_fasta_round_trip(tmp_path, reference):
    path = tmp_path / "germline.fasta"
    write_reference_fasta(reference, path)
    lines = path.read_text().splitlines()
    names = [line[1:] for line in lines[0::2]]
    assert len(names) == 238 and names == sorted(names)
    assert not any(line.startswith(">") for line in lines[1::2])  # one line per sequence
    back = Reference.from_fasta(path)
    assert back.alleles == reference.alleles and back.anchors == {} and back.digest() == reference.digest()

    v, j = sorted(reference.alleles["v"])[0], sorted(reference.alleles["j"])[-1]
    write_reference_fasta(reference, path, alleles=[v, j])  # names sorted: IGHJ... before IGHV...
    assert path.read_text() == f">{j}\n{reference.seq(j)}\n>{v}\n{reference.seq(v)}\n"
    with pytest.raises(ValueError, match=re.escape("IGHVX9*01")):
        write_reference_fasta(reference, path, alleles=[v, "IGHVX9*01"])


def test_reference_from_fasta_parsing(tmp_path):
    path = tmp_path / "tool.fasta"
    # Lowercase and wrapped sequences, and a description after the name.
    path.write_text(">IGHV1-2*02 exported\nacgt\nTTGA\n>IGHJ4*02\nTGGGGC\n")
    assert Reference.from_fasta(path) == Reference(alleles={"v": {"IGHV1-2*02": "ACGTTTGA"},
                                                            "j": {"IGHJ4*02": "TGGGGC"}}, anchors={})
    path.write_text(">IGHV1*01\nACGT\n>IGHC1*01\nACGT\n")
    with pytest.raises(ValueError, match=re.escape("IGHC1*01")):
        Reference.from_fasta(path)
    path.write_text(">IGHV1*01\nACGT\n>IGHV1*01\nACGA\n")
    with pytest.raises(ValueError, match="twice"):
        Reference.from_fasta(path)
    path.write_text("ACGT\n>IGHV1*01\nACGT\n")
    with pytest.raises(ValueError, match="before"):
        Reference.from_fasta(path)


def test_segment_fastas_split(tmp_path, reference):
    paths = write_segment_fastas(reference, tmp_path / "full")
    assert paths == {g: tmp_path / "full" / f"germline_{g}.fasta" for g in "vdj"}
    assert {g: p.read_text().count(">") for g, p in paths.items()} == {"v": 198, "d": 33, "j": 7}
    for g, p in paths.items():
        assert Reference.from_fasta(p).alleles == {g: reference.alleles[g]}

    first = [sorted(reference.alleles[g])[0] for g in "vdj"]
    restricted = write_segment_fastas(reference, tmp_path / "restricted", alleles=first)
    assert [Reference.from_fasta(restricted[g]).alleles[g] for g in "vdj"] == [
        {name: reference.seq(name)} for name in first]
    with pytest.raises(ValueError, match="no d allele"):
        write_segment_fastas(reference, tmp_path / "no_d", alleles=[first[0], first[2]])


def test_verify_reference_names_first_difference(tmp_path, reference):
    path = tmp_path / "tool.fasta"
    write_reference_fasta(reference, path)
    assert verify_reference(path, reference).digest() == reference.digest()

    v = sorted(reference.alleles["v"])
    changed = {**reference.alleles, "v": {**reference.alleles["v"], v[3]: reference.seq(v[3]) + "A"}}
    write_reference_fasta(_without(Reference(changed, {}), v[5]), path)
    with pytest.raises(ValueError) as err:
        verify_reference(path, reference)
    message = str(err.value)
    # The first difference in allele-name order is the changed sequence, before the missing allele.
    assert f"{v[3]} has a different sequence" in message and v[5] not in message
    assert "V=197 D=33 J=7" in message and "V=198 D=33 J=7" in message

    write_reference_fasta(_without(reference, v[0]), path)
    with pytest.raises(ValueError, match=re.escape(f"{v[0]} is missing")):
        verify_reference(path, reference)
    extra = Reference({**reference.alleles, "v": {**reference.alleles["v"], "IGHVX9*01": "ACGT"}}, {})
    write_reference_fasta(extra, path)
    with pytest.raises(ValueError, match=re.escape("IGHVX9*01 is extra")):
        verify_reference(path, reference)


def test_ensure_shared_reference(tmp_path, reference):
    layout = RunLayout(tmp_path / "run")
    assert ensure_shared_reference(layout, reference) == layout.reference_fasta
    assert Reference.from_fasta(layout.reference_fasta).digest() == reference.digest()
    record = json.loads(layout.reference_json.read_text())
    assert record == {"digest": reference.digest(), "counts": {"v": 198, "d": 33, "j": 7}, "source": "truth",
                      "sha256": sha256_file(layout.reference_fasta)}

    # An existing file (for example a tool's exported reference) is verified, not rewritten.
    before = layout.reference_fasta.read_bytes()
    ensure_shared_reference(layout, reference)
    assert layout.reference_fasta.read_bytes() == before
    assert json.loads(layout.reference_json.read_text())["source"] == "existing"

    # A mismatched one is refused and leaves no reference record.
    name = sorted(reference.alleles["d"])[0]
    write_reference_fasta(_without(reference, name), layout.reference_fasta)
    with pytest.raises(ValueError, match=re.escape(f"{name} is missing")):
        ensure_shared_reference(layout, reference)
    assert not layout.reference_json.exists()


# --- timed commands and run records -----------------------------------------------------------------------------

def test_timed_command_success_and_rss(tmp_path):
    log = tmp_path / "logs" / "probe.log"
    code = "import sys; b = bytearray(50_000_000); print('to stdout'); print('to stderr', file=sys.stderr)"
    returncode, wall, rss = timed_command([sys.executable, "-c", code], log_path=log)
    assert returncode == 0 and wall > 0 and rss > 40_000
    assert "to stdout" in log.read_text() and "to stderr" in log.read_text()


def test_timed_command_env_and_cpus(tmp_path):
    log = tmp_path / "probe.log"
    code = "import os; print(os.environ['ASSAY_PROBE'], sorted(os.sched_getaffinity(0)))"
    returncode, _, _ = timed_command([sys.executable, "-c", code], log_path=log, env={"ASSAY_PROBE": "seen"},
                                     cpus="0")
    assert returncode == 0 and log.read_text().strip() == "seen [0]"


def test_timed_command_failure(tmp_path):
    returncode, wall, rss = timed_command([sys.executable, "-c", "raise SystemExit(3)"], log_path=tmp_path / "x.log")
    assert returncode == 3 and wall > 0 and rss > 0
    returncode, _, _ = timed_command([str(tmp_path / "no_such_tool")], log_path=tmp_path / "y.log")
    assert returncode != 0


def test_failed_run_is_refused(tmp_path):
    result = RunResult(tool="igblast", suite="S1", arm="shared", predictions=str(tmp_path / "predictions.tsv"),
                       n_reads=10, wall_seconds=1.5, peak_rss_kb=None, returncode=3,
                       commands=[["igblastn", "-query", "reads.fasta"]], provenance={"version": "1.22.0"})
    path = tmp_path / "run" / "run.json"
    write_run_json(result, path)
    assert read_run_json(path) == result
    with pytest.raises(ValueError, match=r"igblast on S1/shared failed with return code 3"):
        require_successful_run(path)

    succeeded = dataclasses.replace(result, returncode=0)
    write_run_json(succeeded, path)
    assert require_successful_run(path) == succeeded
    with pytest.raises(FileNotFoundError, match="no run record"):
        require_successful_run(tmp_path / "missing" / "run.json")


# --- registry and oracle ----------------------------------------------------------------------------------------

def test_runner_options_paths_are_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    options = RunnerOptions(igblast_dir="igblast", alignair_checkpoint="~/m.alignair", cache_dir="cache")
    assert options.igblast_dir == str(tmp_path / "igblast") and options.cache_dir == str(tmp_path / "cache")
    assert options.alignair_checkpoint == str(Path.home() / "m.alignair")
    assert RunnerOptions().igblast_dir is None and RunnerOptions().cache_dir is None


def test_registry(tmp_path):
    assert {"igblast", "oracle", "alignair", "alignair-local"} <= set(RUNNERS)
    assert isinstance(make_runner("oracle", RunnerOptions()), OracleRunner)
    runner = make_runner("igblast", RunnerOptions(igblast_dir=str(tmp_path), threads=2, cpus="0-1"))
    assert isinstance(runner, IgBlastRunner) and runner.name == "igblast"
    assert (runner.igblast_dir, runner.threads, runner.cpus) == (tmp_path, 2, "0-1")
    with pytest.raises(ValueError, match="--igblast-dir"):
        make_runner("igblast", RunnerOptions())
    with pytest.raises(ValueError, match="unknown tool 'nope'.*igblast"):
        make_runner("nope", RunnerOptions())


def test_oracle_run_scores_perfect(run_layout, reference, tmp_path):
    runner = OracleRunner()
    assert runner.prepare(run_layout, reference) == {}
    out = tmp_path / "oracle.tsv"
    result = runner.run(run_layout.reads("S0"), out, suite="S0", arm="shared", log_dir=tmp_path / "logs")
    assert (result.tool, result.suite, result.arm, result.predictions) == ("oracle", "S0", "shared", str(out))
    assert result.returncode == 0 and result.n_reads == 50 and result.commands == []
    assert result.provenance["truth"] == {"path": str(run_layout.truth("S0")),
                                          "sha256": sha256_file(run_layout.truth("S0"))}
    truths = read_truth_jsonl(run_layout.truth("S0"))
    scores, matching = score_tool(truths, out, reference)
    assert matching["rows"] == 50 and matching["missing"] == 0
    assert scores == assert_oracle_perfect(truths, reference)


def test_oracle_writes_exactly_the_reads_it_is_given(run_layout, tmp_path):
    lines = run_layout.reads("S0").read_text().splitlines()
    ids = [line[1:] for line in lines[0::2]]
    fasta = tmp_path / "some.fasta"
    fasta.write_text("".join(f"{lines[i]}\n{lines[i + 1]}\n" for i in (8, 2, 4)))
    runner = OracleRunner(run_layout)  # bound at construction, without prepare
    result = runner.run(fasta, tmp_path / "some.tsv", suite="S0", arm="genotype", log_dir=tmp_path / "logs",
                        alleles=["ignored*01"])
    assert result.n_reads == 3
    assert [p for p in load_airr_tsv(tmp_path / "some.tsv")[0]] == [ids[4], ids[1], ids[2]]

    fasta.write_text(">S0:plain:999999\nACGT\n")
    with pytest.raises(ValueError, match="S0:plain:999999"):
        runner.run(fasta, tmp_path / "bad.tsv", suite="S0", arm="shared", log_dir=tmp_path / "logs")
    with pytest.raises(RuntimeError, match="prepare"):
        OracleRunner().run(fasta, tmp_path / "bad.tsv", suite="S0", arm="shared", log_dir=tmp_path / "logs")


# --- the run command --------------------------------------------------------------------------------------------

def test_cli_run_oracle(run_layout, reference, capsys):
    assert _run(run_layout, "oracle") == 0
    result = require_successful_run(run_layout.run_json("oracle", "S0", "shared"))
    assert result.n_reads == 50 and result.predictions == str(run_layout.predictions("oracle", "S0", "shared"))
    assert result.provenance["reads"] == {"path": str(run_layout.reads("S0")),
                                          "sha256": sha256_file(run_layout.reads("S0"))}
    shared = json.loads(run_layout.reference_json.read_text())
    assert (shared["digest"], shared["source"]) == (reference.digest(), "truth")

    truths = read_truth_jsonl(run_layout.truth("S0"))
    scores, _ = score_tool(truths, run_layout.predictions("oracle", "S0", "shared"), reference)
    assert scores == assert_oracle_perfect(truths, reference)
    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == 1 and printed[0].startswith("oracle S0/shared: return code 0, 50 reads")


def test_cli_run_records_absolute_paths(run_layout, monkeypatch):
    monkeypatch.chdir(run_layout.root.parent)
    assert main(["run", "--run-dir", run_layout.root.name, "--tool", "oracle", "--suites", "S0"]) == 0
    result = read_run_json(run_layout.run_json("oracle", "S0", "shared"))
    assert result.predictions == str(run_layout.predictions("oracle", "S0", "shared"))
    assert Path(result.predictions).is_absolute() and Path(result.provenance["reads"]["path"]).is_absolute()


def test_cli_run_refuses_mismatched_reference(run_layout, reference, capsys):
    name = sorted(reference.alleles["v"])[0]
    run_layout.reference_dir.mkdir()
    write_reference_fasta(_without(reference, name), run_layout.reference_fasta)  # e.g. another model's export
    assert _run(run_layout, "oracle") == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and f"{name} is missing" in err
    assert not (run_layout.root / "runs").exists()  # refused before running


def test_cli_run_input_errors(run_layout, tmp_path, capsys):
    assert _run(run_layout, "nope") == 2
    assert "unknown tool 'nope'" in capsys.readouterr().err
    assert main(["run", "--run-dir", str(run_layout.root), "--tool", "oracle", "--suites", "S0,S1"]) == 2
    assert "S1 is not built" in capsys.readouterr().err
    assert _run(run_layout, "igblast", "--igblast-dir", str(tmp_path / "no_igblast")) == 2
    assert "igblastn" in capsys.readouterr().err
    assert not (run_layout.root / "runs").exists()


def test_cli_run_checks_the_truth_reference(run_layout, capsys):
    header, rows = read_header(run_layout.truth("S0")), read_jsonl(run_layout.truth("S0"))
    shutil.copytree(run_layout.suite_dir("S0"), run_layout.suite_dir("S0B"))
    write_jsonl(rows, run_layout.truth("S0B"), header={**header, "reference": "HUMAN_IGK_OGRDB"})
    assert main(["run", "--run-dir", str(run_layout.root), "--tool", "oracle", "--suites", "S0,S0B"]) == 2
    assert "built against different references" in capsys.readouterr().err

    write_jsonl(rows, run_layout.truth("S0"), header={**header, "reference_digest": "0" * 64})
    assert _run(run_layout, "oracle") == 2
    assert "HUMAN_IGH_OGRDB now has digest" in capsys.readouterr().err
    assert not run_layout.reference_dir.exists()


class _FailingRunner:
    """Writes a partial TSV, then exits with status 5."""

    name = "failing"

    def prepare(self, layout, reference):
        return {}

    def run(self, fasta, out_tsv, *, suite, arm, log_dir, alleles=None):
        Path(out_tsv).write_text("sequence_id\tsequence\nS0:plain:000000\tAC")
        argv = [sys.executable, "-c", "raise SystemExit(5)"]
        returncode, wall, rss = timed_command(argv, log_path=Path(log_dir) / "failing.log")
        return RunResult(self.name, suite, arm, str(out_tsv), 50, wall, rss, returncode, [argv], {})


class _CrashingRunner:
    name = "failing"

    def prepare(self, layout, reference):
        return {}

    def run(self, fasta, out_tsv, *, suite, arm, log_dir, alleles=None):
        raise RuntimeError("runner crashed")


def test_cli_run_records_failed_run(run_layout, monkeypatch, capsys):
    monkeypatch.setitem(RUNNERS, "failing", lambda options: _FailingRunner())
    assert _run(run_layout, "failing") == 2
    run_json = run_layout.run_json("failing", "S0", "shared")
    assert read_run_json(run_json).returncode == 5
    with pytest.raises(ValueError, match="failing on S0/shared failed with return code 5"):
        require_successful_run(run_json)
    err = capsys.readouterr().err
    assert "failing on S0/shared failed with return code 5" in err

    # A run that dies without a result leaves no run record, so an earlier one cannot be mistaken for it; the
    # command reports the error (no traceback) and exits with status 2.
    write_run_json(dataclasses.replace(read_run_json(run_json), returncode=0), run_json)
    monkeypatch.setitem(RUNNERS, "failing", lambda options: _CrashingRunner())
    assert _run(run_layout, "failing") == 2
    assert capsys.readouterr().err == "error: runner crashed\n"
    assert not run_json.exists()


# --- IgBLAST ----------------------------------------------------------------------------------------------------

def test_igblast_prepare_needs_the_release(run_layout, reference, tmp_path):
    with pytest.raises(FileNotFoundError, match="igblastn"):
        IgBlastRunner(tmp_path / "no_igblast").prepare(run_layout, reference)


def _fake_release(directory: Path) -> Path:
    """A stand-in IgBLAST release: ``igblastn`` prints a version, and ``makeblastdb`` succeeds except on a FASTA
    under a ``db`` directory (the restricted databases), where it fails with status 3."""
    (directory / "bin").mkdir(parents=True)
    (directory / "optional_file").mkdir()
    (directory / "optional_file" / "human_gl.aux").write_text("aux\n")
    scripts = {"igblastn": 'echo "igblastn: 1.22.0"\n',
               "makeblastdb": 'case "$*" in */db/*) echo "makeblastdb: cannot build" >&2; exit 3;; esac\n'}
    for name, body in scripts.items():
        path = directory / "bin" / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o755)
    return directory


def test_igblast_fingerprint(tmp_path):
    release = _fake_release(tmp_path / "release")
    runner = IgBlastRunner(release)
    fingerprint = runner.fingerprint()
    assert fingerprint == {"igblast_dir": str(release), "igblastn_sha256": sha256_file(release / "bin" / "igblastn"),
                           "makeblastdb_sha256": sha256_file(release / "bin" / "makeblastdb"),
                           "auxiliary_data_sha256": sha256_file(release / "optional_file" / "human_gl.aux")}
    # An upgrade in place keeps the directory but changes the fingerprint.
    (release / "bin" / "igblastn").write_text('#!/bin/sh\necho "igblastn: 1.23.0"\n')
    assert runner.fingerprint()["igblastn_sha256"] != fingerprint["igblastn_sha256"]
    with pytest.raises(FileNotFoundError, match="igblastn"):
        IgBlastRunner(tmp_path / "no_igblast").fingerprint()
    assert OracleRunner().fingerprint() == {}


def test_igblast_restricted_database_failure_is_recorded(run_layout, reference, tmp_path):
    runner = IgBlastRunner(_fake_release(tmp_path / "release"), threads=1)
    runner.prepare(run_layout, reference)
    v, d, j = (sorted(reference.alleles[g])[:2] for g in "vdj")
    # A genotype with no D allele, and a makeblastdb that fails, are failed runs, not exceptions.
    for alleles, expected, name in (([*v, *j], "no d allele", "no_d"), ([*v, *d, *j], "makeblastdb failed", "fails")):
        log_dir = tmp_path / name
        result = runner.run(run_layout.reads("S0"), tmp_path / f"{name}.tsv", suite="S0", arm="genotype",
                            log_dir=log_dir, alleles=alleles)
        assert result.returncode != 0 and result.n_reads == 50 and result.commands == []
        assert expected in result.provenance["error"] and expected in (log_dir / "igblastn.log").read_text()
        assert result.provenance["version"] == "igblastn: 1.22.0"


@needs_igblast
def test_igblast_on_toy_suite(run_layout, reference):
    assert _run(run_layout, "igblast", "--igblast-dir", IGBLAST_DIR, "--threads", "2") == 0
    result = require_successful_run(run_layout.run_json("igblast", "S0", "shared"))
    assert result.n_reads == 50 and result.wall_seconds > 0 and result.peak_rss_kb > 0

    release, db = Path(IGBLAST_DIR), run_layout.work_dir("igblast")
    out = run_layout.predictions("igblast", "S0", "shared")
    assert result.commands == [[
        "env", f"IGDATA={release}", str(release / "bin" / "igblastn"),
        "-germline_db_V", str(db / "V"), "-germline_db_D", str(db / "D"), "-germline_db_J", str(db / "J"),
        "-auxiliary_data", str(release / "optional_file" / "human_gl.aux"),
        "-organism", "human", "-ig_seqtype", "Ig", "-num_threads", "2", "-outfmt", "19",
        "-query", str(run_layout.reads("S0")), "-out", str(out)]]
    prov = result.provenance
    assert prov["version"].startswith("igblastn: 1.22.0")
    assert prov["sha256"] == {"igblastn": sha256_file(release / "bin" / "igblastn"),
                              "makeblastdb": sha256_file(release / "bin" / "makeblastdb"),
                              "auxiliary_data": sha256_file(release / "optional_file" / "human_gl.aux")}
    assert (prov["threads"], prov["cpus"]) == (2, None)
    files = prov["database"]["files"]
    assert {"germline_v.fasta", "V.nsq", "D.nsq", "J.nsq"} <= set(files)
    assert files["germline_v.fasta"] == sha256_file(db / "germline_v.fasta")
    assert prov["database"]["alleles"] == {"v": 198, "d": 33, "j": 7}
    assert [c[-1] for c in prov["database"]["commands"]] == [str(db / x) for x in "VDJ"]

    truths = read_truth_jsonl(run_layout.truth("S0"))
    scores, matching = score_tool(truths, out, reference)
    assert matching["rows"] == 50 and matching["missing"] == 0
    v = [case["v.top1_in_set"] for case in scores.values() if "v.top1_in_set" in case]
    assert sum(v) / len(v) > 0.8


@needs_igblast
def test_igblast_restricted_to_alleles(run_layout, reference, tmp_path):
    truths = read_truth_jsonl(run_layout.truth("S0"))
    v_alleles = sorted({a for t in truths for a in t.segments["v"].alleles.alleles})
    runner = IgBlastRunner(Path(IGBLAST_DIR), threads=2)
    runner.prepare(run_layout, reference)
    log_dir = tmp_path / "logs"
    result = runner.run(run_layout.reads("S0"), tmp_path / "out.tsv", suite="S0", arm="genotype", log_dir=log_dir,
                        alleles=[*v_alleles, *reference.alleles["d"], *reference.alleles["j"]])
    assert result.returncode == 0
    assert result.provenance["database"]["alleles"] == {"v": len(v_alleles), "d": 33, "j": 7}
    db = log_dir / "db"
    assert Reference.from_fasta(db / "germline_v.fasta").alleles["v"].keys() == set(v_alleles)
    # The restricted database is built for this run, then searched.
    makeblastdb = str(Path(IGBLAST_DIR) / "bin" / "makeblastdb")
    assert [(c[0], c[-1]) for c in result.commands[:3]] == [(makeblastdb, str(db / x)) for x in "VDJ"]
    assert result.commands[3][3:5] == ["-germline_db_V", str(db / "V")]
    assert result.provenance["database"]["commands"] == result.commands[:3]
    calls = {a for p in load_airr_tsv(tmp_path / "out.tsv")[0].values() for a in p.segments["v"].call_set}
    assert calls and calls <= set(v_alleles)


@needs_igblast
def test_igblast_missing_db_records_failure(run_layout, monkeypatch, capsys):
    class WithoutVDatabase(IgBlastRunner):
        def prepare(self, layout, reference):
            provenance = super().prepare(layout, reference)
            for path in layout.work_dir(self.name).glob("V.*"):
                path.unlink()
            return provenance

    monkeypatch.setitem(RUNNERS, "igblast",
                        lambda options: WithoutVDatabase(Path(options.igblast_dir), threads=options.threads))
    assert _run(run_layout, "igblast", "--igblast-dir", IGBLAST_DIR) == 2
    run_json = run_layout.run_json("igblast", "S0", "shared")
    assert read_run_json(run_json).returncode != 0
    with pytest.raises(ValueError, match="igblast on S0/shared failed"):
        require_successful_run(run_json)
    assert "igblast on S0/shared failed" in capsys.readouterr().err
