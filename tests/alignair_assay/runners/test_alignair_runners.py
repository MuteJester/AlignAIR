"""AlignAIR runners (the Docker image and a local checkpoint) and the genotype arm.

The image tests run only with ``ALIGNAIR_ASSAY_DOCKER=1`` (Docker and the AlignAIR image available); the local
checkpoint tests only when ``ALIGNAIR_ASSAY_MODEL`` names a ``.alignair`` model file.
"""
import dataclasses
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("GenAIRR")

import alignair_assay  # noqa: E402
from alignair_assay.airr.reader import load_airr_tsv  # noqa: E402
from alignair_assay.cli import main  # noqa: E402
from alignair_assay.core.cases import read_cases, write_cases  # noqa: E402
from alignair_assay.core.layout import RunLayout  # noqa: E402
from alignair_assay.core.provenance import git_state, sha256_file, tree_digest  # noqa: E402
from alignair_assay.core.reference import Reference  # noqa: E402
from alignair_assay.runners import RUNNERS, make_runner  # noqa: E402
from alignair_assay.runners.alignair import (  # noqa: E402
    CLI_ENTRY,
    DEFAULT_IMAGE,
    DEFAULT_MODEL,
    MEASURED_SCRIPT,
    AlignAIRImage,
    AlignAIRLocal,
    local_code_identity,
    memory_peak_kb,
    write_genotype_json,
)
from alignair_assay.runners import alignair as alignair_module  # noqa: E402
from alignair_assay.runners.arms import merge_tsvs, run_genotype_arm  # noqa: E402
from alignair_assay.runners.base import (  # noqa: E402
    RunnerOptions,
    fasta_ids,
    read_run_json,
    require_successful_run,
)
from alignair_assay.runners.oracle import OracleRunner  # noqa: E402
from alignair_assay.runners.reference import verify_reference  # noqa: E402
from alignair_assay.scoring.pipeline import score_tool  # noqa: E402
from alignair_assay.suites.donors import donors_from_json  # noqa: E402
from alignair_assay.testing import assert_oracle_perfect  # noqa: E402
from alignair_assay.truth.io import read_truth_jsonl  # noqa: E402
from alignair_assay.workflow.build import build_suite  # noqa: E402

pytestmark = pytest.mark.genairr_build  # these tests build GenAIRR suites
DOCKER = os.environ.get("ALIGNAIR_ASSAY_DOCKER") == "1"
MODEL = os.environ.get("ALIGNAIR_ASSAY_MODEL")
needs_docker = pytest.mark.skipif(not DOCKER, reason="set ALIGNAIR_ASSAY_DOCKER=1 to run the AlignAIR image")
needs_model = pytest.mark.skipif(not MODEL, reason="set ALIGNAIR_ASSAY_MODEL to a .alignair model file")
SOURCE_DIR = Path(alignair_assay.__file__).resolve().parent.parent
# AlignAIR's ``predict --genotype`` passes ``load_genotype``'s (genotype, unknown) tuple on as the genotype.
GENOTYPE_CRASH = "'tuple' object has no attribute 'items'"


@pytest.fixture(scope="module")
def cohort_run(tmp_path_factory):
    """S1 at scale 0.02 (100 cases over donors donor00-donor09; donor10-donor19 have none), built once."""
    root = tmp_path_factory.mktemp("cohort")
    build_suite("S1", RunLayout(root), reference_name="HUMAN_IGH_OGRDB", split="dev", seed=5, scale=0.02, workers=4)
    return root


@pytest.fixture
def cohort_layout(cohort_run, tmp_path):
    """A private copy of the built S1 run directory."""
    root = tmp_path / "run"
    shutil.copytree(cohort_run, root)
    return RunLayout(root)


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory):
    """S0 at scale 1.0 (50 cases), built once (only for the tests that run a real AlignAIR)."""
    root = tmp_path_factory.mktemp("toy")
    build_suite("S0", RunLayout(root), reference_name="HUMAN_IGH_OGRDB", split="dev", seed=5, scale=1.0, workers=1)
    return root


@pytest.fixture
def toy_layout(toy_run, tmp_path):
    root = tmp_path / "toy"
    shutil.copytree(toy_run, root)
    return RunLayout(root)


def _donors(layout: RunLayout, suite: str = "S1"):
    return donors_from_json(json.loads(layout.donors(suite).read_text()))


def _reads_by_subject(layout: RunLayout, suite: str = "S1") -> dict[str, list[str]]:
    subject = {case["case_id"]: case["groups"]["subject"] for case in read_cases(layout.cases(suite))}
    by_subject: dict[str, list[str]] = {}
    for read_id in fasta_ids(layout.reads(suite)):
        by_subject.setdefault(subject[read_id], []).append(read_id)
    return by_subject


class RecordingOracle(OracleRunner):
    """Oracle rows for the reads of each FASTA it is given; records each call's read ids and alleles, and reports
    1 s of wall time and a peak RSS of 100 KB times the call number."""

    name = "fake"

    def __init__(self, layout):
        super().__init__(layout)
        self.calls = []

    def run(self, fasta, out_tsv, *, suite, arm, log_dir, alleles=None):
        self.calls.append((fasta_ids(fasta), sorted(alleles)))
        result = super().run(fasta, out_tsv, suite=suite, arm=arm, log_dir=log_dir, alleles=alleles)
        return dataclasses.replace(result, tool=self.name, wall_seconds=1.0, peak_rss_kb=100 * len(self.calls),
                                   commands=[["fake", str(fasta)]])


# --- command lines ----------------------------------------------------------------------------------------------

def test_image_command_line(tmp_path):
    runner = AlignAIRImage("alignair/alignair:9.9@sha256:abc", cache_dir=tmp_path / "cache", threads=2, cpus="0-1")
    fasta, out = tmp_path / "in" / "reads.fasta", tmp_path / "out" / "predictions.tsv"
    user = f"{os.getuid()}:{os.getgid()}"
    predict = ["predict", "--model", DEFAULT_MODEL, "--input", "/in/reads.fasta", "--out", "/out/predictions.tsv",
               "--device", "cpu", "--quiet"]
    assert runner.predict_command(fasta, out) == [
        "docker", "run", "--rm", "-u", user, "-v", f"{tmp_path / 'cache'}:/home/appuser/.cache/alignair",
        "-v", f"{fasta.parent}:/in:ro", "-v", f"{out.parent}:/out", "--cpuset-cpus", "0-1",
        "-e", "OMP_NUM_THREADS=2", "-e", "MKL_NUM_THREADS=2",
        "--entrypoint", "sh", "alignair/alignair:9.9@sha256:abc", "-c", MEASURED_SCRIPT, "sh", *predict]
    # cat's own error never reaches the log, so it cannot become the recorded error line.
    assert MEASURED_SCRIPT == 'alignair "$@" 2>&1; rc=$?; cat /sys/fs/cgroup/memory.peak 2>/dev/null; exit $rc'

    # The genotype file is read from the FASTA's directory, mounted at /in.
    with_genotype = runner.predict_command(fasta, out, genotype=fasta.parent / "genotype.json")
    assert with_genotype[-2:] == ["--genotype", "/in/genotype.json"]
    with pytest.raises(ValueError, match="/in"):
        runner.predict_command(fasta, out, genotype=tmp_path / "elsewhere" / "genotype.json")

    # Without threads or CPUs there is no pinning; a plain call uses the image's entrypoint.
    plain = AlignAIRImage("img", model="m@2", cache_dir=tmp_path / "cache")
    assert plain.docker_command(["models", "path", "m@2"]) == [
        "docker", "run", "--rm", "-u", user, "-v", f"{tmp_path / 'cache'}:/home/appuser/.cache/alignair",
        "img", "models", "path", "m@2"]
    with pytest.raises(RuntimeError, match="cache"):
        AlignAIRImage("img").docker_command(["doctor"])

    # The registry factory: the pinned default image and model, the run's thread count and CPUs.
    built = make_runner("alignair", RunnerOptions(threads=3, cpus="4-5", cache_dir="relative/cache"))
    assert isinstance(built, AlignAIRImage) and built.name == "alignair"
    assert (built.image, built.model, built.threads, built.cpus) == (DEFAULT_IMAGE, DEFAULT_MODEL, 3, "4-5")
    assert built.cache_dir == Path.cwd() / "relative" / "cache"
    assert DEFAULT_IMAGE.startswith("alignair/alignair:3.0.0@sha256:5df66a34")


def test_memory_peak_parsing(tmp_path):
    log = tmp_path / "alignair.log"
    log.write_text("aligned 3 reads (0 dropped) -> /out/x.tsv\n358596608\n")
    assert memory_peak_kb(log) == 358596608 // 1024
    log.write_text("aligned 3 reads\ncat: can't open '/sys/fs/cgroup/memory.peak': No such file or directory\n")
    assert memory_peak_kb(log) is None
    log.write_text("")
    assert memory_peak_kb(log) is None
    assert memory_peak_kb(tmp_path / "missing.log") is None


def test_image_run_mounts_only_the_reads(cohort_layout, reference, tmp_path, monkeypatch):
    """The container's /in holds the reads FASTA alone (and the genotype JSON in the genotype arm), never the
    suite directory with its cases and truth."""
    seen = []

    def fake_timed_command(argv, *, log_path, env=None, cpus=None):
        mounted = Path(next(arg for arg in argv if arg.endswith(":/in:ro")).removesuffix(":/in:ro"))
        seen.append((mounted, sorted(path.name for path in mounted.iterdir())))
        Path(log_path).write_text("aligned\n358596608\n")
        return 0, 0.5, None

    monkeypatch.setattr(alignair_module, "timed_command", fake_timed_command)
    runner = AlignAIRImage("img", cache_dir=tmp_path / "cache")
    runner._setup, runner._reference = {}, reference  # what prepare would record
    log_dir = cohort_layout.logs_dir("alignair", "S1", "shared")
    (log_dir / "input").mkdir(parents=True)
    (log_dir / "input" / "stale.json").write_text("{}")
    result = runner.run(cohort_layout.reads("S1"), cohort_layout.predictions("alignair", "S1", "shared"), suite="S1",
                        arm="shared", log_dir=log_dir)
    assert result.returncode == 0 and result.peak_rss_kb == 358596608 // 1024
    assert seen == [(log_dir / "input", ["reads.fasta"])]
    assert (log_dir / "input" / "reads.fasta").read_bytes() == cohort_layout.reads("S1").read_bytes()
    assert result.commands == [runner.predict_command(log_dir / "input" / "reads.fasta",
                                                      cohort_layout.predictions("alignair", "S1", "shared"))]

    genotype_logs = tmp_path / "genotype_logs"
    alleles = [sorted(reference.alleles[g])[0] for g in "vdj"]
    runner.run(cohort_layout.reads("S1"), tmp_path / "out" / "predictions.tsv", suite="S1", arm="genotype",
               log_dir=genotype_logs, alleles=alleles)
    assert seen[-1] == (genotype_logs / "input", ["genotype.json", "reads.fasta"])


def test_merge_writes_each_case_once(tmp_path):
    header = "sequence_id\tv_call\n"
    first, second = tmp_path / "donor00.tsv", tmp_path / "donor01.tsv"
    first.write_text(header + "S1:a\tIGHV1-2*02\nS1:b\tIGHV3-23*01\nS1:a\tIGHV1-69*01\n\n")
    second.write_text(header + "S1:c\tIGHV4-34*01\nS1:c\tIGHV4-34*02")
    out = tmp_path / "merged.tsv"
    # A read the tool reports twice is written once: its first row, as the AIRR reader keeps it.
    assert merge_tsvs([("donor00", first, ["S1:a", "S1:b"]), ("donor01", second, ["S1:c"])], out) == 3
    assert out.read_text() == header + "S1:a\tIGHV1-2*02\nS1:b\tIGHV3-23*01\nS1:c\tIGHV4-34*01\n"


def test_local_command_line(tmp_path):
    checkpoint = tmp_path / "model.alignair"
    runner = AlignAIRLocal(checkpoint, device="cuda", batch_size=32, threads=4, cpus="2")
    fasta, out, genotype = tmp_path / "reads.fasta", tmp_path / "out.tsv", tmp_path / "logs" / "genotype.json"
    assert runner.predict_command(fasta, out) == [
        sys.executable, "-c", "import sys; from alignair.cli.main import main; sys.exit(main())",
        "predict", "--model", str(checkpoint), "--input", str(fasta), "--out", str(out), "--device", "cuda",
        "--batch-size", "32", "--quiet"]
    assert CLI_ENTRY == "import sys; from alignair.cli.main import main; sys.exit(main())"
    assert runner.predict_command(fasta, out, genotype=genotype)[-2:] == ["--genotype", str(genotype)]
    assert runner.environment() == {"PYTHONPATH": str(SOURCE_DIR), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    assert AlignAIRLocal(checkpoint).environment() == {"PYTHONPATH": str(SOURCE_DIR)}

    built = make_runner("alignair-local", RunnerOptions(alignair_checkpoint="models/x.alignair", device="cpu",
                                                        batch_size=16, threads=2))
    assert isinstance(built, AlignAIRLocal) and built.name == "alignair-local"
    assert (built.checkpoint, built.device, built.batch_size, built.threads, built.cpus) == (
        Path.cwd() / "models" / "x.alignair", "cpu", 16, 2, None)
    with pytest.raises(ValueError, match="--checkpoint"):
        make_runner("alignair-local", RunnerOptions())
    with pytest.raises(FileNotFoundError, match=re.escape(str(checkpoint))):
        runner.prepare(RunLayout(tmp_path / "run"), None)
    with pytest.raises(FileNotFoundError, match=re.escape(str(checkpoint))):
        runner.fingerprint()
    checkpoint.write_bytes(b"weights")
    assert runner.fingerprint() == {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                                    "device": "cuda", **local_code_identity()}
    assert {"alignair", "alignair-local", "igblast", "oracle"} <= set(RUNNERS)


def test_local_fingerprint_covers_the_code_it_runs(tmp_path, monkeypatch):
    # The subprocess runs this checkout's src/alignair: its sources are part of what the runner predicts with.
    identity = local_code_identity()
    assert identity == {"code_sha256": tree_digest(SOURCE_DIR / "alignair", suffixes=(".py", ".pyx")),
                        "torch_version": importlib.metadata.version("torch")}
    source = tmp_path / "checkout" / "src"
    shutil.copytree(SOURCE_DIR / "alignair", source / "alignair", ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(alignair_module, "SOURCE_DIR", source)
    checkpoint = tmp_path / "model.alignair"
    checkpoint.write_bytes(b"weights")
    runner = AlignAIRLocal(checkpoint)
    before = runner.fingerprint()
    assert before["code_sha256"] == identity["code_sha256"]  # the same files, wherever they are
    assert "git_commit" not in before and "git_dirty" not in before  # the checkout's state is provenance only
    # A decoder or post-processing edit, the checkpoint unchanged: another tool.
    predict = source / "alignair" / "cli" / "predict.py"
    predict.write_text(predict.read_text() + "\n# decoder change\n")
    after = runner.fingerprint()
    assert after["code_sha256"] != before["code_sha256"]
    assert {key: value for key, value in after.items() if key != "code_sha256"} == {
        key: value for key, value in before.items() if key != "code_sha256"}
    # Byte-code caches and other files are not code.
    (source / "alignair" / "cli" / "__pycache__").mkdir(exist_ok=True)
    (source / "alignair" / "cli" / "__pycache__" / "predict.cpython-312.pyc").write_bytes(b"\0")
    (source / "alignair" / "notes.md").write_text("n\n")
    assert runner.fingerprint() == after


def test_unrelated_commit_keeps_local_fingerprint(tmp_path, monkeypatch):
    # The same sources at another commit (and a dirty tree elsewhere in the repository): the same fingerprint.
    source = tmp_path / "checkout" / "src"
    shutil.copytree(SOURCE_DIR / "alignair", source / "alignair", ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(alignair_module, "SOURCE_DIR", source)
    checkpoint = tmp_path / "model.alignair"
    checkpoint.write_bytes(b"weights")
    runner = AlignAIRLocal(checkpoint)
    states = iter([{"git_commit": "a" * 40, "git_dirty": False}, {"git_commit": "b" * 40, "git_dirty": True}])
    monkeypatch.setattr(alignair_module, "git_state", lambda directory: next(states), raising=False)
    first, second = runner.fingerprint(), runner.fingerprint()
    assert first == second and set(local_code_identity()) == {"code_sha256", "torch_version"}


def test_tool_genotype_written(cohort_layout, reference, tmp_path):
    donor = _donors(cohort_layout)[0]
    genotype = donor.tool_genotype(reference)
    alleles = [name for names in genotype.values() for name in names]
    path = tmp_path / "genotype.json"
    # Keys are the gene letters AlignAIR's load_genotype reads; each list is sorted.
    assert write_genotype_json(reference, reversed(alleles), path) == genotype
    assert json.loads(path.read_text()) == genotype and list(genotype) == ["v", "d", "j"]
    assert all(names == sorted(names) and names for names in genotype.values())

    with pytest.raises(ValueError, match=re.escape("IGHVX9*01")):
        write_genotype_json(reference, [*alleles, "IGHVX9*01"], tmp_path / "bad.json")
    with pytest.raises(ValueError, match="no d allele"):
        write_genotype_json(reference, genotype["v"] + genotype["j"], tmp_path / "no_d.json")
    assert not (tmp_path / "bad.json").exists() and not (tmp_path / "no_d.json").exists()


# --- the genotype arm -------------------------------------------------------------------------------------------

def test_genotype_arm_merges_each_case_once(cohort_layout, reference):
    runner = RecordingOracle(cohort_layout)
    result = run_genotype_arm(runner, cohort_layout, "S1", reference, tool="fake")

    reads = fasta_ids(cohort_layout.reads("S1"))
    by_subject = _reads_by_subject(cohort_layout)
    donors = {donor.subject_id: donor for donor in _donors(cohort_layout)}
    # One run per donor with cases, in donor order, on exactly its reads (suite order), restricted to its genotype.
    assert [ids for ids, _ in runner.calls] == [by_subject[s] for s in sorted(by_subject)]
    for (ids, alleles), subject in zip(runner.calls, sorted(by_subject)):
        genotype = donors[subject].tool_genotype(reference)
        assert alleles == sorted(name for names in genotype.values() for name in names)
        assert fasta_ids(cohort_layout.donor_reads("fake", "S1", "genotype", subject)) == ids

    merged = cohort_layout.predictions("fake", "S1", "genotype")
    predictions, stats = load_airr_tsv(merged)
    assert stats == {"rows": 100, "duplicates": 0, "invalid_rows": 0} and sorted(predictions) == sorted(reads)
    assert merged.read_text().count("sequence_id") == 1  # one header
    truths = read_truth_jsonl(cohort_layout.truth("S1"))
    scores, matching = score_tool(truths, merged, reference)
    assert matching["missing"] == 0 and scores == assert_oracle_perfect(truths, reference)

    record = require_successful_run(cohort_layout.run_json("fake", "S1", "genotype"))
    assert record == result
    assert (record.tool, record.suite, record.arm, record.predictions) == ("fake", "S1", "genotype", str(merged))
    assert record.n_reads == 100 and record.returncode == 0
    assert record.wall_seconds == pytest.approx(10.0) and record.peak_rss_kb == 1000  # summed; the largest
    assert record.commands == [["fake", str(cohort_layout.donor_reads("fake", "S1", "genotype", s))]
                               for s in sorted(by_subject)]
    prov = record.provenance
    assert prov["reads"] == {"path": str(cohort_layout.reads("S1")), "sha256": sha256_file(cohort_layout.reads("S1"))}
    assert prov["donors_json"] == {"path": str(cohort_layout.donors("S1")),
                                   "sha256": sha256_file(cohort_layout.donors("S1"))}
    assert sorted(prov["donors"]) == sorted(by_subject)
    first = prov["donors"]["donor00"]
    assert first["reads"] == len(by_subject["donor00"]) and first["returncode"] == 0
    assert first["alleles"] == {g: len(n) for g, n in donors["donor00"].tool_genotype(reference).items()}
    assert "truth" in first["provenance"]


def test_genotype_arm_skips_empty_donor(cohort_layout, reference):
    runner = RecordingOracle(cohort_layout)
    result = run_genotype_arm(runner, cohort_layout, "S1", reference)  # the run directory defaults to runner.name
    empty = [f"donor{i:02d}" for i in range(10, 20)]
    assert result.provenance["skipped_donors"] == empty
    assert len(runner.calls) == 10 and not set(empty) & set(result.provenance["donors"])
    for subject in empty:
        assert not cohort_layout.donor_reads("fake", "S1", "genotype", subject).exists()
        assert not cohort_layout.donor_predictions("fake", "S1", "genotype", subject).exists()
    assert sum(len(ids) for ids, _ in runner.calls) == result.n_reads == 100


def test_genotype_arm_header_mismatch_raises(cohort_layout, reference):
    class OtherHeader(RecordingOracle):
        def run(self, fasta, out_tsv, **kwargs):
            result = super().run(fasta, out_tsv, **kwargs)
            if "donor03" in Path(fasta).name:
                lines = Path(out_tsv).read_text().splitlines()
                Path(out_tsv).write_text("\n".join([lines[0] + "\textra", *(line + "\t" for line in lines[1:])]) + "\n")
            return result

    with pytest.raises(ValueError, match="donor03.*header"):
        run_genotype_arm(OtherHeader(cohort_layout), cohort_layout, "S1", reference)
    assert not cohort_layout.run_json("fake", "S1", "genotype").exists()
    assert not cohort_layout.predictions("fake", "S1", "genotype").exists()

    class ForeignRow(RecordingOracle):
        def run(self, fasta, out_tsv, **kwargs):
            result = super().run(fasta, out_tsv, **kwargs)
            if "donor05" in Path(fasta).name:  # a row for a read of another donor
                lines = Path(out_tsv).read_text().splitlines()
                Path(out_tsv).write_text("\n".join([*lines, lines[1].replace("S1:", "S1:x", 1)]) + "\n")
            return result

    with pytest.raises(ValueError, match="donor05.*not one of its reads"):
        run_genotype_arm(ForeignRow(cohort_layout), cohort_layout, "S1", reference)


def test_genotype_arm_records_failed_donor(cohort_layout, reference):
    class FailsOnDonor03(RecordingOracle):
        def run(self, fasta, out_tsv, **kwargs):
            result = super().run(fasta, out_tsv, **kwargs)
            if "donor03" not in Path(fasta).name:
                return result
            return dataclasses.replace(result, returncode=4, provenance={**result.provenance, "error": "boom"})

    runner = FailsOnDonor03(cohort_layout)
    result = run_genotype_arm(runner, cohort_layout, "S1", reference)
    assert result.returncode == 4 and len(runner.calls) == 4  # donors after the failed one are not run
    # The record names the failed donor and the tool's error line.
    assert (result.provenance["failed_donor"], result.provenance["error"]) == ("donor03", "boom")
    assert result.provenance["donors"]["donor03"]["returncode"] == 4
    run_json = cohort_layout.run_json("fake", "S1", "genotype")
    assert read_run_json(run_json).returncode == 4
    with pytest.raises(ValueError, match="fake on S1/genotype failed with return code 4"):
        require_successful_run(run_json)
    assert not cohort_layout.predictions("fake", "S1", "genotype").exists()


def test_genotype_arm_input_errors(cohort_layout, reference):
    cases = read_cases(cohort_layout.cases("S1"))
    write_cases([{**cases[0], "groups": {}}, *cases[1:]], cohort_layout.cases("S1"))
    with pytest.raises(ValueError, match=re.escape(f"{cases[0]['case_id']} has no subject")):
        run_genotype_arm(RecordingOracle(cohort_layout), cohort_layout, "S1", reference)
    write_cases([{**cases[0], "groups": {"subject": "donor99"}}, *cases[1:]], cohort_layout.cases("S1"))
    with pytest.raises(ValueError, match="donor99 is not a donor"):
        run_genotype_arm(RecordingOracle(cohort_layout), cohort_layout, "S1", reference)
    cohort_layout.donors("S1").unlink()
    with pytest.raises(FileNotFoundError, match="donors"):
        run_genotype_arm(RecordingOracle(cohort_layout), cohort_layout, "S1", reference)


def test_cli_genotype_arm(cohort_layout, reference, monkeypatch, capsys):
    monkeypatch.setitem(RUNNERS, "fake", lambda options: RecordingOracle(None))
    args = ["run", "--run-dir", str(cohort_layout.root), "--tool", "fake", "--suites", "S1", "--arm", "genotype"]
    assert main(args) == 0
    record = require_successful_run(cohort_layout.run_json("fake", "S1", "genotype"))
    assert record.n_reads == 100 and len(record.provenance["donors"]) == 10
    assert capsys.readouterr().out.startswith("fake S1/genotype: return code 0, 100 reads")

    # A suite without donors is refused before anything runs.
    build_suite("S0", cohort_layout, reference_name="HUMAN_IGH_OGRDB", split="dev", seed=5, scale=0.1, workers=1)
    shutil.rmtree(cohort_layout.root / "runs")
    assert main([*args[:6], "S1,S0", *args[7:]]) == 2
    assert "suite S0 has no donors" in capsys.readouterr().err
    assert not (cohort_layout.root / "runs").exists()


# --- the AlignAIR image -----------------------------------------------------------------------------------------

def _restricted_v_alleles(layout: RunLayout, reference: Reference) -> list[str]:
    truths = read_truth_jsonl(layout.truth("S0"))
    return sorted({a for t in truths for a in t.segments["v"].alleles.alleles})


def _check_restricted_run(runner, layout, reference, tmp_path):
    """Run ``runner`` on S0 restricted to the truth V alleles: a bad genotype is a recorded failure, and the V
    calls stay inside the genotype (xfail while AlignAIR's ``predict --genotype`` crashes)."""
    v_alleles = _restricted_v_alleles(layout, reference)
    bad = runner.run(layout.reads("S0"), tmp_path / "bad.tsv", suite="S0", arm="genotype",
                     log_dir=tmp_path / "bad_logs", alleles=[*v_alleles, "IGHVX9*01"])
    assert bad.returncode != 0 and "IGHVX9*01" in bad.provenance["error"]
    assert "IGHVX9*01" in (tmp_path / "bad_logs" / "alignair.log").read_text()

    log_dir, out = tmp_path / "logs", tmp_path / "restricted.tsv"
    alleles = [*v_alleles, *reference.alleles["d"], *reference.alleles["j"]]
    result = runner.run(layout.reads("S0"), out, suite="S0", arm="genotype", log_dir=log_dir, alleles=alleles)
    genotype = result.provenance["genotype"]
    assert json.loads(Path(genotype["path"]).read_text())["v"] == v_alleles
    assert genotype["alleles"] == {"v": len(v_alleles), "d": 33, "j": 7}
    if result.returncode != 0:
        # A failed run is recorded with the tool's error line, which is also in its log.
        assert result.provenance["error"] and result.provenance["error"] in (log_dir / "alignair.log").read_text()
        if GENOTYPE_CRASH in result.provenance["error"]:
            pytest.xfail("alignair predict --genotype crashes: it passes load_genotype's (genotype, unknown) "
                         "tuple on")
    assert result.returncode == 0
    calls = {a for p in load_airr_tsv(out)[0].values() for a in p.segments["v"].call_set}
    assert calls and calls <= set(v_alleles)


def _check_scores(layout, tool, reference):
    truths = read_truth_jsonl(layout.truth("S0"))
    scores, matching = score_tool(truths, layout.predictions(tool, "S0", "shared"), reference)
    assert matching["rows"] == 50 and matching["missing"] == 0
    v = [case["v.top1_in_set"] for case in scores.values() if "v.top1_in_set" in case]
    assert sum(v) / len(v) > 0.8


@needs_docker
def test_image_on_toy_suite(toy_layout, reference, tmp_path):
    assert main(["run", "--run-dir", str(toy_layout.root), "--tool", "alignair", "--suites", "S0",
                 "--threads", "2"]) == 0
    result = require_successful_run(toy_layout.run_json("alignair", "S0", "shared"))
    assert result.n_reads == 50 and result.wall_seconds > 0 and result.peak_rss_kb > 100_000
    prov = result.provenance
    inspected = json.loads(subprocess.run(["docker", "image", "inspect", DEFAULT_IMAGE], capture_output=True,
                                          text=True, check=True).stdout)[0]
    assert prov["image"] == {"reference": DEFAULT_IMAGE, "id": inspected["Id"],
                             "repo_digests": inspected["RepoDigests"]}
    assert prov["model"]["spec"] == DEFAULT_MODEL
    assert prov["model"]["sha256"] == prov["sidecar"]["artifact_sha256"]
    assert prov["cache"]["dir"] == str(toy_layout.work_dir("alignair") / "cache")
    assert prov["cache"]["model_source"] in ("host_cache", "registry")
    assert "alleles:  V=198 D=33 J=7" in prov["model_card"] and prov["doctor"]["alignair"] == "3.0.0"
    assert prov["model_reference"]["digest"] == reference.digest()
    assert (prov["threads"], prov["cpus"]) == (2, None) and "rss_note" not in prov
    assert prov["sidecar"]["counts"]["input_records"] == 50
    run_dir = toy_layout.run_dir("alignair", "S0", "shared")
    assert not (run_dir / "predictions.tsv.run.json").exists()
    assert (toy_layout.logs_dir("alignair", "S0", "shared") / "alignair.run.json").is_file()
    # The container sees a staged copy of the reads, not the suite directory.
    staged = toy_layout.logs_dir("alignair", "S0", "shared") / "input" / "reads.fasta"
    assert result.commands == [AlignAIRImage(DEFAULT_IMAGE, cache_dir=prov["cache"]["dir"], threads=2)
                               .predict_command(staged, toy_layout.predictions("alignair", "S0", "shared"))]
    _check_scores(toy_layout, "alignair", reference)

    # A second preparation finds the model in the run's cache.
    runner = make_runner("alignair", RunnerOptions(threads=2))
    assert runner.prepare(toy_layout, reference)["cache"]["model_source"] == "cache_dir"
    # The fingerprint resolves what the image and model references point at.
    assert runner.fingerprint() == {"image": DEFAULT_IMAGE, "image_id": prov["image"]["id"], "model": DEFAULT_MODEL,
                                    "model_sha256": prov["model"]["sha256"]}


@needs_docker
def test_image_restricted_to_alleles(toy_layout, reference, tmp_path):
    runner = make_runner("alignair", RunnerOptions(threads=2))
    runner.prepare(toy_layout, reference)
    _check_restricted_run(runner, toy_layout, reference, tmp_path)


@needs_docker
def test_exported_reference_matches_truth(tmp_path):
    runner = AlignAIRImage(DEFAULT_IMAGE, cache_dir=tmp_path / "cache")
    exported = runner.export_reference(tmp_path / "out" / "model_reference.fasta")
    truth = Reference.from_dataconfig("HUMAN_IGH_OGRDB")
    assert verify_reference(exported, truth).digest() == truth.digest()


# --- a local checkpoint -----------------------------------------------------------------------------------------

@needs_model
def test_local_cpu_on_toy_suite(toy_layout, reference, tmp_path):
    checkpoint = Path(MODEL).expanduser()
    assert main(["run", "--run-dir", str(toy_layout.root), "--tool", "alignair-local", "--suites", "S0",
                 "--checkpoint", str(checkpoint), "--device", "cpu", "--batch-size", "32", "--threads", "4"]) == 0
    result = require_successful_run(toy_layout.run_json("alignair-local", "S0", "shared"))
    assert result.n_reads == 50 and result.wall_seconds > 0 and result.peak_rss_kb > 100_000
    prov = result.provenance
    assert prov["checkpoint"] == {"path": str(checkpoint.absolute()), "sha256": sha256_file(checkpoint)}
    assert prov["repository"]["path"] == str(SOURCE_DIR.parent) and prov["repository"]["git_commit"]
    assert prov["torch"] and prov["cuda_device"] is None and prov["device"] == "cpu"
    assert prov["model_reference"]["digest"] == reference.digest()
    assert prov["sidecar"]["counts"]["input_records"] == 50 and prov["threads"] == 4
    reads, predictions = toy_layout.reads("S0"), toy_layout.predictions("alignair-local", "S0", "shared")
    runner = AlignAIRLocal(checkpoint, device="cpu", batch_size=32, threads=4)
    assert result.commands == [["env", f"PYTHONPATH={SOURCE_DIR}", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4",
                                *runner.predict_command(reads, predictions)]]
    _check_scores(toy_layout, "alignair-local", reference)


@needs_model
def test_local_restricted_to_alleles(toy_layout, reference, tmp_path):
    runner = AlignAIRLocal(Path(MODEL).expanduser(), device="cpu", batch_size=32, threads=4)
    runner.prepare(toy_layout, reference)
    _check_restricted_run(runner, toy_layout, reference, tmp_path)
