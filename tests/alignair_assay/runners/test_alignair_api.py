"""The alignair-api runner: AlignAIR's public Python API run on a checkpoint, in a subprocess.

The command-line, fingerprint and failure tests run offline (the subprocess monkeypatched); the end-to-end test runs
the real subprocess on a small random-init model saved as a ``.alignair`` file. The equivalence with the CLI runner
(alignair-local) runs only when ``ALIGNAIR_ASSAY_MODEL`` names a ``.alignair`` model file, and on the GPU too with
``ALIGNAIR_ASSAY_GPU=1``. The genotype arm on real W-97 reads also needs ``ALIGNAIR_ASSAY_DATA`` (the PRJNA555323
data root), ``ALIGNAIR_ASSAY_DOCKER=1`` (its captured genes) and Task 4's run in ``.private/assay/p3a_task4_run``.
"""
import csv
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("GenAIRR")

import alignair_assay  # noqa: E402
from alignair_assay.cli import main  # noqa: E402
from alignair_assay.core.cases import evenly_spaced, read_fasta  # noqa: E402
from alignair_assay.core.layout import RunLayout  # noqa: E402
from alignair_assay.core.provenance import sha256_file  # noqa: E402
from alignair_assay.core.reference import Reference  # noqa: E402
from alignair_assay.runners import RUNNERS, make_runner  # noqa: E402
from alignair_assay.runners import alignair_api as api_module  # noqa: E402
from alignair_assay.runners.alignair import local_code_identity  # noqa: E402
from alignair_assay.runners.alignair_api import (  # noqa: E402
    DRIVER,
    DRIVER_MODULE,
    GENOTYPE_API_FAILED,
    GENOTYPE_NOTE,
    AlignAIRApi,
)
from alignair_assay.runners.base import RunnerOptions, error_line, require_successful_run  # noqa: E402
from alignair_assay.workflow.build import build_suite  # noqa: E402
from tests.alignair_assay.stored_runs import RUNS  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.genairr_build]  # these tests build GenAIRR suites
MODEL = os.environ.get("ALIGNAIR_ASSAY_MODEL")
GPU = os.environ.get("ALIGNAIR_ASSAY_GPU") == "1"
DATA = os.environ.get("ALIGNAIR_ASSAY_DATA")
DOCKER = os.environ.get("ALIGNAIR_ASSAY_DOCKER") == "1"
RUN = RUNS / "p3a_task4_run"
needs_model = pytest.mark.skipif(not MODEL, reason="set ALIGNAIR_ASSAY_MODEL to a .alignair model file")
needs_gpu = pytest.mark.skipif(not GPU, reason="set ALIGNAIR_ASSAY_GPU=1")
SOURCE_DIR = Path(alignair_assay.__file__).resolve().parent.parent


def _fasta(path: Path, reads: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f">{name}\n{sequence}\n" for name, sequence in reads.items()), encoding="utf-8")
    return path


def _rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or ()), list(reader)


def _v_calls(path: Path) -> list[set[str]]:
    return [{name for name in row["v_call"].split(",") if name} for row in _rows(path)[1]]


# --- command line and fingerprint -------------------------------------------------------------------------------

def test_api_command_line(tmp_path):
    checkpoint = tmp_path / "model.alignair"
    runner = AlignAIRApi(checkpoint, device="cuda", batch_size=32, threads=4, cpus="2")
    fasta, out, genotype = tmp_path / "reads.fasta", tmp_path / "out.tsv", tmp_path / "logs" / "genotype.json"
    reference = tmp_path / "work" / "model_reference.fasta"
    assert runner.predict_command(fasta, out, reference=reference) == [
        sys.executable, "-m", "alignair_assay.diagnostics.api_predict", "--checkpoint", str(checkpoint),
        "--reference", str(reference), "--input", str(fasta), "--out", str(out), "--device", "cuda",
        "--batch-size", "32"]
    assert runner.predict_command(fasta, out, genotype, reference=reference)[-2:] == ["--genotype", str(genotype)]
    # The module the subprocess runs is the diagnostics layer's (which may import alignair), found by name here
    # without importing it.
    assert DRIVER_MODULE == "alignair_assay.diagnostics.api_predict"
    assert DRIVER == SOURCE_DIR / "alignair_assay" / "diagnostics" / "api_predict.py" and DRIVER.is_file()
    assert Path(importlib.util.find_spec(DRIVER_MODULE).origin) == DRIVER
    assert runner.environment() == {"PYTHONPATH": str(SOURCE_DIR), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}

    built = make_runner("alignair-api", RunnerOptions(alignair_checkpoint="models/x.alignair", device="cpu",
                                                      batch_size=16, threads=2))
    assert isinstance(built, AlignAIRApi) and built.name == "alignair-api" and "alignair-api" in RUNNERS
    assert (built.checkpoint, built.device, built.batch_size, built.threads, built.cpus) == (
        Path.cwd() / "models" / "x.alignair", "cpu", 16, 2, None)
    with pytest.raises(ValueError, match="alignair-api runner needs .*--checkpoint"):
        make_runner("alignair-api", RunnerOptions())
    with pytest.raises(FileNotFoundError, match=re.escape(str(checkpoint))):
        runner.prepare(RunLayout(tmp_path / "run"), None)
    with pytest.raises(FileNotFoundError, match=re.escape(str(checkpoint))):
        runner.fingerprint()
    checkpoint.write_bytes(b"weights")
    # alignair-local's fingerprint (the checkpoint, its digest, the device, the src/alignair digest and torch
    # version), and the module that drives the API in their place of the CLI.
    assert runner.fingerprint() == {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                                    "device": "cuda", **local_code_identity(), "driver_sha256": sha256_file(DRIVER)}


def test_api_fingerprint_covers_the_driver(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.alignair"
    checkpoint.write_bytes(b"weights")
    driver = tmp_path / "api_predict.py"
    shutil.copyfile(DRIVER, driver)
    monkeypatch.setattr(api_module, "DRIVER", driver)
    runner = AlignAIRApi(checkpoint)
    before = runner.fingerprint()
    driver.write_text(driver.read_text() + "\n# a reader or writer change\n")
    after = runner.fingerprint()
    assert after["driver_sha256"] != before["driver_sha256"]
    assert {k: v for k, v in after.items() if k != "driver_sha256"} == {
        k: v for k, v in before.items() if k != "driver_sha256"}


# --- runs, the subprocess monkeypatched --------------------------------------------------------------------------

def _unprepared_run(tmp_path: Path, reference: Reference) -> AlignAIRApi:
    """A runner as ``prepare`` leaves it, without running anything."""
    runner = AlignAIRApi(tmp_path / "model.alignair", device="cpu", batch_size=8, threads=2)
    runner._setup = {"model_reference": {"path": str(tmp_path / "work" / "model_reference.fasta")}}
    runner._reference = reference
    return runner


READS = {"r1": "ACGT" * 50, "r2": "TTGA" * 50, "r3": "GGCA" * 50}


def test_runner_records_failed_genotype_path(reference, tmp_path, monkeypatch):
    calls = []
    failure = "genotype API path failed: InputError: the genotype leaves gene V with no allele"

    def fake_timed_command(argv, *, log_path, env=None, cpus=None):
        calls.append((argv, env, cpus))
        Path(log_path).write_text("Traceback (most recent call last):\n  File \"api_predict.py\", line 1\n"
                                  "alignair.errors.exceptions.InputError: the genotype leaves gene V with no allele\n"
                                  f"{failure}\n")
        return GENOTYPE_API_FAILED, 0.5, 1234

    monkeypatch.setattr(api_module, "timed_command", fake_timed_command)
    runner = _unprepared_run(tmp_path, reference)
    fasta = _fasta(tmp_path / "reads.fasta", READS)
    out, logs = tmp_path / "out" / "predictions.tsv", tmp_path / "logs"
    alleles = [sorted(reference.alleles[letter])[0] for letter in "vdj"]
    result = runner.run(fasta, out, suite="real", arm="genotype", log_dir=logs, alleles=alleles)

    assert GENOTYPE_API_FAILED == 3 and GENOTYPE_NOTE == "genotype API path failed"
    assert (result.tool, result.suite, result.arm, result.returncode) == ("alignair-api", "real", "genotype", 3)
    assert result.provenance["note"] == GENOTYPE_NOTE and result.provenance["error"] == failure
    assert (result.n_reads, result.wall_seconds, result.peak_rss_kb) == (3, 0.5, 1234)
    (argv, env, cpus), = calls
    genotype = logs / "genotype.json"
    assert argv == runner.predict_command(fasta, out, genotype, reference=tmp_path / "work" / "model_reference.fasta")
    assert json.loads(genotype.read_text()) == {"v": [alleles[0]], "d": [alleles[1]], "j": [alleles[2]]}
    assert result.provenance["genotype"] == {"path": str(genotype), "sha256": sha256_file(genotype),
                                             "alleles": {"v": 1, "d": 1, "j": 1}}
    assert env == runner.environment() and cpus is None
    assert result.commands == [["env", *(f"{k}={v}" for k, v in runner.environment().items()), *argv]]
    assert result.provenance["sidecar"] is None and result.provenance["threads"] == 2

    # A failure without a genotype is the tool's error alone, never the genotype path's.
    def fake_shared_failure(argv, *, log_path, env=None, cpus=None):
        Path(log_path).write_text("Traceback (most recent call last):\nRuntimeError: CUDA out of memory\n")
        return 1, 0.2, None

    monkeypatch.setattr(api_module, "timed_command", fake_shared_failure)
    shared = runner.run(fasta, out, suite="real", arm="shared", log_dir=tmp_path / "shared_logs")
    assert shared.returncode == 1 and shared.provenance["error"] == "RuntimeError: CUDA out of memory"
    assert "note" not in shared.provenance and shared.provenance["genotype"] is None


def test_runner_records_a_genotype_it_cannot_write_without_running(reference, tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "timed_command", lambda *a, **k: pytest.fail("the command must not run"))
    runner = _unprepared_run(tmp_path, reference)
    fasta = _fasta(tmp_path / "reads.fasta", READS)
    bad = runner.run(fasta, tmp_path / "out.tsv", suite="real", arm="genotype", log_dir=tmp_path / "logs",
                     alleles=[*(sorted(reference.alleles[g])[0] for g in "vdj"), "IGHVX9*01"])
    assert bad.returncode == 1 and "IGHVX9*01" in bad.provenance["error"] and "note" not in bad.provenance
    empty = runner.run(fasta, tmp_path / "out.tsv", suite="real", arm="genotype", log_dir=tmp_path / "logs2",
                       alleles=[sorted(reference.alleles["v"])[0]])
    assert empty.returncode == 1 and "no d allele" in empty.provenance["error"]
    with pytest.raises(RuntimeError, match="not prepared"):
        AlignAIRApi(tmp_path / "model.alignair").run(fasta, tmp_path / "out.tsv", suite="real", arm="shared",
                                                     log_dir=tmp_path / "logs3")


def test_runner_keeps_the_subprocess_record(reference, tmp_path, monkeypatch):
    record = {"counts": {"input_records": 3, "written_records": 3}, "device": "cpu"}

    def fake_timed_command(argv, *, log_path, env=None, cpus=None):
        out = Path(argv[argv.index("--out") + 1])
        out.write_text("sequence_id\nr1\nr2\nr3\n")
        Path(f"{out}.run.json").write_text(json.dumps(record))
        Path(log_path).write_text("")
        return 0, 1.5, 2048

    monkeypatch.setattr(api_module, "timed_command", fake_timed_command)
    runner = _unprepared_run(tmp_path, reference)
    out, logs = tmp_path / "out" / "predictions.tsv", tmp_path / "logs"
    out.parent.mkdir()
    Path(f"{out}.run.json").write_text("{\"stale\": true}")
    result = runner.run(_fasta(tmp_path / "reads.fasta", READS), out, suite="real", arm="shared", log_dir=logs)
    assert result.returncode == 0 and "error" not in result.provenance and "note" not in result.provenance
    # The record moves next to the logs, as alignair-local's predict sidecar does.
    assert result.provenance["sidecar"] == record and not Path(f"{out}.run.json").exists()
    assert json.loads((logs / "api_predict.run.json").read_text()) == record


# --- the real subprocess on a small model ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_checkpoint(tmp_path_factory, make_small_model):
    """A small random-init model on HUMAN_IGH_OGRDB, written as a ``.alignair`` model file."""
    from alignair.model_file import save_model

    model, _ = make_small_model("HUMAN_IGH_OGRDB", seed=1)
    path = tmp_path_factory.mktemp("model") / "small.alignair"
    save_model(str(path), model, dataconfigs=["HUMAN_IGH_OGRDB"], training={}, include_trusted_pickle=False)
    return path


def test_runner_end_to_end_on_a_small_model(small_checkpoint, reference, tmp_path):
    names = {letter: sorted(reference.alleles[letter]) for letter in "vdj"}
    reads = {f"r{i}": "".join(reference.seq(names[g][k]) for g, k in (("v", i * 19), ("d", i), ("j", i % 7)))
             for i in range(8)}
    fasta = _fasta(tmp_path / "reads.fasta", reads)
    layout = RunLayout(tmp_path / "run")
    runner = AlignAIRApi(small_checkpoint, device="cpu", batch_size=4, threads=2)
    setup = runner.prepare(layout, reference)
    assert setup["model_reference"]["digest"] == reference.digest()
    assert setup["model_reference"]["path"] == str(layout.work_dir("alignair-api") / "model_reference.fasta")
    assert setup["driver"] == {"module": DRIVER_MODULE, "path": str(DRIVER), "sha256": sha256_file(DRIVER)}

    shared = runner.run(fasta, tmp_path / "shared.tsv", suite="toy", arm="shared", log_dir=tmp_path / "shared_logs")
    assert shared.returncode == 0, shared.provenance.get("error")
    assert [row["sequence_id"] for row in _rows(tmp_path / "shared.tsv")[1]] == list(reads)
    assert shared.provenance["sidecar"]["counts"]["written_records"] == 8 and shared.peak_rss_kb > 0

    allowed = names["v"][:5]
    restricted = runner.run(fasta, tmp_path / "genotype.tsv", suite="toy", arm="genotype",
                            log_dir=tmp_path / "genotype_logs", alleles=[*allowed, *names["d"], *names["j"]])
    assert restricted.returncode == 0, restricted.provenance.get("error")
    calls = _v_calls(tmp_path / "genotype.tsv")
    assert any(calls) and all(call <= set(allowed) for call in calls)
    assert restricted.provenance["sidecar"]["genotype"]["alleles"] == {"v": 5, "d": 33, "j": 7}

    # The subprocess tells a failure on the genotype path by its exit status, with the note as the log's error
    # line. The runner never writes a genotype the API refuses (an empty letter), so the driver is run directly.
    genotype = tmp_path / "refused" / "genotype.json"
    genotype.parent.mkdir()
    genotype.write_text(json.dumps({"v": [], "d": names["d"], "j": names["j"]}))
    log = tmp_path / "refused" / "alignair.log"
    argv = runner.predict_command(fasta, tmp_path / "refused.tsv", genotype,
                                  reference=Path(setup["model_reference"]["path"]))
    returncode, _, _ = api_module.timed_command(argv, log_path=log, env=runner.environment())
    assert returncode == GENOTYPE_API_FAILED
    assert error_line(log).startswith(f"{GENOTYPE_NOTE}: InputError: the genotype leaves gene V with no allele")
    assert not (tmp_path / "refused.tsv").exists() and list(tmp_path.glob("refused.tsv*")) == []


# --- the real checkpoint ------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def s1_run(tmp_path_factory):
    """S1 at scale 0.02 (100 cases), built once, as the other runner tests build it."""
    root = tmp_path_factory.mktemp("s1")
    build_suite("S1", RunLayout(root), reference_name="HUMAN_IGH_OGRDB", split="dev", seed=5, scale=0.02, workers=4)
    return root


def _differences(header: list[str], cli: list[dict], api: list[dict]) -> dict[str, list[str]]:
    """Per field, the reads whose value differs between the two runs' rows (read by read, in file order)."""
    return {field: [a["sequence_id"] for a, b in zip(cli, api) if a[field] != b[field]] for field in header
            if any(a[field] != b[field] for a, b in zip(cli, api))}


@needs_model
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=needs_gpu)])
def test_api_rows_equal_the_cli_rows(s1_run, reference, tmp_path, device):
    """On the same reads and checkpoint, the API path writes the CLI's rows, field by field: one product code."""
    root = tmp_path / "run"
    shutil.copytree(s1_run, root)
    layout = RunLayout(root)
    for tool in ("alignair-local", "alignair-api"):
        assert main(["run", "--run-dir", str(root), "--tool", tool, "--suites", "S1", "--checkpoint", MODEL,
                     "--device", device, "--batch-size", "32", "--threads", "4"]) == 0
    cli_header, cli = _rows(layout.predictions("alignair-local", "S1", "shared"))
    api_header, api = _rows(layout.predictions("alignair-api", "S1", "shared"))
    assert api_header == cli_header
    assert [row["sequence_id"] for row in api] == [row["sequence_id"] for row in cli]
    assert len(api) == 100
    differing = _differences(cli_header, cli, api)
    assert not differing, {field: (len(ids), ids[:3]) for field, ids in differing.items()}
    # and the files are the same bytes, line endings and column order included
    assert (layout.predictions("alignair-api", "S1", "shared").read_bytes()
            == layout.predictions("alignair-local", "S1", "shared").read_bytes())

    result = require_successful_run(layout.run_json("alignair-api", "S1", "shared"))
    local = require_successful_run(layout.run_json("alignair-local", "S1", "shared"))
    prov = result.provenance
    assert prov["checkpoint"] == local.provenance["checkpoint"] and prov["device"] == device
    assert prov["model_reference"]["digest"] == reference.digest() == local.provenance["model_reference"]["digest"]
    assert prov["driver"] == {"module": DRIVER_MODULE, "path": str(DRIVER), "sha256": sha256_file(DRIVER)}
    # The subprocess's record, like the CLI's sidecar, counts every read written.
    assert prov["sidecar"]["counts"]["written_records"] == local.provenance["sidecar"]["counts"]["written_records"]
    assert prov["sidecar"]["device"] == device and prov["sidecar"]["genotype"] is None


@needs_model
@pytest.mark.skipif(not DATA or not DOCKER or not RUN.is_dir(),
                    reason="set ALIGNAIR_ASSAY_DATA to the PRJNA555323 data root and ALIGNAIR_ASSAY_DOCKER=1, with "
                           "Task 4's run in .private/assay/p3a_task4_run")
def test_api_genotype_arm_on_real_w97_reads(reference, tmp_path):
    """W-97's genotype arm on 50 of its real reads: the API path runs on the real checkpoint, and every V call is
    one of the donor's arm alleles."""
    from alignair_assay.real.genotype import IuisMap, arm_alleles, load_vdjbase_csv
    from alignair_assay.real.locus import donor_capture

    root = Path(DATA)
    samples = root / "genotypes/vdjbase_genomic_samples"
    iuis = IuisMap.from_ogrdb(root / "genotypes/ogrdb_human_IGH_VDJ/Homo_sapiens_IGH_VDJ_rev_8.json", reference)
    genotype = donor_capture(load_vdjbase_csv(samples / "P25_I151_S1.csv", "W-97", reference, iuis),
                             samples / "P25_I151_S1.bam", root / "genotypes/igenotyper/gene_coords.bed", iuis)
    alleles = arm_alleles(genotype, reference, iuis)
    allowed_v = {name for name in alleles if reference.gene_letter(name) == "v"}
    reads = dict(evenly_spaced(read_fasta(RunLayout(RUN).real_reads("W-97")), 50))
    fasta = _fasta(tmp_path / "reads.fasta", reads)

    runner = AlignAIRApi(Path(MODEL).expanduser(), device="cuda" if GPU else "cpu", batch_size=64, threads=4)
    runner.prepare(RunLayout(tmp_path / "run"), reference)
    out = tmp_path / "genotype.tsv"
    result = runner.run(fasta, out, suite="real", arm="genotype", log_dir=tmp_path / "logs", alleles=alleles)
    assert result.returncode == 0, (result.provenance.get("note"), result.provenance.get("error"))
    assert result.provenance["genotype"]["alleles"] == {"v": len(allowed_v), "d": 33, "j": 7}
    calls = _v_calls(out)
    assert len(calls) == 50 and sum(bool(call) for call in calls) >= 45
    outside = [call - allowed_v for call in calls if call - allowed_v]
    assert not outside, outside
