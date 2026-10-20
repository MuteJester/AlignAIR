"""What every runner shares: the runner interface, the record of one run, timed commands and run records.

A runner turns a FASTA of reads into an AIRR rearrangement TSV. ``prepare`` does the tool's one-time setup for a
run directory; ``run`` annotates one FASTA and returns a ``RunResult``, which ``run_suite`` stores as the run's
``run.json``. Orchestration, the genotype arm and the speed protocol use nothing but this interface.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from alignair_assay.core.cases import read_fasta
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import sha256_file
from alignair_assay.core.reference import Reference

# GNU time: ``-f %M`` writes the command's peak resident set size in KB.
TIME = "/usr/bin/time"


@dataclass(frozen=True)
class RunResult:
    """One tool run on one FASTA: the ``predictions`` TSV path, the number of reads given (``n_reads``), the wall
    time in seconds, the peak resident set size in KB (None when not measured), the tool's return code (0 is
    success; anything else means the predictions must not be scored), every command executed, and provenance."""

    tool: str
    suite: str
    arm: str
    predictions: str
    n_reads: int
    wall_seconds: float
    peak_rss_kb: int | None
    returncode: int
    commands: list[list[str]]
    provenance: dict


@dataclass(frozen=True)
class RunnerOptions:
    """Tool settings, one set for every runner; each runner reads the fields it needs. ``threads`` is the
    tool's thread count and ``cpus`` a ``taskset``/cpuset CPU list (``"0-7"``) to pin it to. The path fields
    (``igblast_dir``, ``alignair_checkpoint``, ``cache_dir``) are made absolute, ``~`` expanded, so the commands
    a run records can be re-run from anywhere."""

    igblast_dir: str | None = None
    threads: int = 8
    cpus: str | None = None
    alignair_image: str | None = None
    alignair_model: str = "alignair-igh-human@1.0.0"
    alignair_checkpoint: str | None = None
    device: str = "cpu"
    batch_size: int = 64
    cache_dir: str | None = None

    def __post_init__(self) -> None:
        for name in ("igblast_dir", "alignair_checkpoint", "cache_dir"):
            value = getattr(self, name)
            if value:
                object.__setattr__(self, name, absolute_path(value))


class Runner(Protocol):
    """A tool under test.

    ``prepare(layout, reference)`` does the one-time setup for run directory ``layout`` against the shared
    reference (already verified equal to the truth reference) and returns its provenance; it raises when the
    tool cannot be set up. ``run(fasta, out_tsv, ...)`` then annotates the reads of ``fasta`` into the AIRR TSV
    ``out_tsv``, keeping logs and scratch files in ``log_dir``; ``alleles``, when given, restricts the tool to
    those reference alleles (the genotype arm). A tool that fails, or a restriction it cannot take, returns a
    ``RunResult`` with a non-zero return code and the reason as the provenance's ``error`` rather than raising, so
    the failure is recorded. ``fingerprint()`` identifies what the tool predicts with, by content as well as by
    name (binary digests, the resolved image ID, the model file's digest), without preparing a run directory; a
    run made with another fingerprint is another tool's run.
    """

    name: str

    def fingerprint(self) -> dict: ...

    def prepare(self, layout: RunLayout, reference: Reference) -> dict: ...

    def run(self, fasta: Path, out_tsv: Path, *, suite: str, arm: str, log_dir: Path,
            alleles: Iterable[str] | None = None) -> RunResult: ...


def timed_command(argv, *, log_path: Path, env: dict[str, str] | None = None,
                  cpus: str | None = None) -> tuple[int, float, int | None]:
    """Run ``argv`` and return ``(return code, wall seconds, peak RSS in KB)``.

    The command runs under GNU time (``/usr/bin/time -f %M``), and under ``taskset -c <cpus>`` when ``cpus`` is
    given; ``env`` adds variables to the current environment. Its stdout and stderr go to ``log_path`` (its
    directory is created), and the peak RSS, None when it cannot be read, to ``<log_path>.peak_rss_kb``.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rss_path = log_path.with_name(log_path.name + ".peak_rss_kb")
    rss_path.unlink(missing_ok=True)
    command = [str(arg) for arg in argv]
    if cpus is not None:
        command = ["taskset", "-c", str(cpus), *command]
    command = [TIME, "-f", "%M", "-o", str(rss_path), *command]
    with open(log_path, "wb") as log:
        start = time.perf_counter()
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})})
        wall = time.perf_counter() - start
    return completed.returncode, wall, _peak_rss_kb(rss_path)


def absolute_path(path) -> str:
    """``path`` with ``~`` expanded, made absolute against the current directory and normalised (no symlinks
    resolved)."""
    return os.path.abspath(os.path.expanduser(str(path)))


def fasta_ids(fasta) -> list[str]:
    """The read ids of a FASTA file, in file order."""
    return [name for name, _ in read_fasta(fasta)]


def failed_run(tool: str, *, suite: str, arm: str, fasta: Path, out_tsv: Path, log_path: Path, message: str,
               provenance: dict | None = None) -> RunResult:
    """The record of a run that failed before its tool could run, for example on a genotype the tool cannot be
    restricted to: return code 1, no wall time (0.0 s), no peak RSS and no command. ``message`` is written to
    ``log_path`` (its directory is created) and kept as the provenance's ``error``."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(message + "\n", encoding="utf-8")
    return RunResult(tool=tool, suite=suite, arm=arm, predictions=str(out_tsv), n_reads=len(fasta_ids(fasta)),
                     wall_seconds=0.0, peak_rss_kb=None, returncode=1, commands=[],
                     provenance={**(provenance or {}), "error": message})


def error_line(log_path: Path) -> str | None:
    """The line of a failed command's log that says why it failed: the last non-empty line that is not a bare
    number (a peak-memory readout), such as the exception line ending a Python traceback; None without one."""
    try:
        lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if line and not line.isdigit():
            return line
    return None


def write_run_json(result: RunResult, path: Path) -> None:
    """Write a run record as strict JSON (its directory is created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_run_json(path: Path) -> RunResult:
    """Inverse of ``write_run_json``."""
    return RunResult(**json.loads(Path(path).read_text(encoding="utf-8")))


def require_successful_run(run_json: Path) -> RunResult:
    """The run record at ``run_json`` when its run succeeded. A run with a non-zero return code raises ValueError
    naming the tool, suite, arm and return code; a missing record (a run that never finished) raises
    FileNotFoundError. Nothing may score a run that does not pass this check."""
    path = Path(run_json)
    if not path.is_file():
        raise FileNotFoundError(f"no run record {path}: the run never finished, so it cannot be scored")
    result = read_run_json(path)
    if result.returncode != 0:
        raise ValueError(f"{result.tool} on {result.suite}/{result.arm} failed with return code {result.returncode}; "
                         f"its predictions are not scored (see {path})")
    return result


def run_suite(runner: Runner, layout: RunLayout, suite: str, *, arm: str = "shared",
              alleles: Iterable[str] | None = None, settings: dict | None = None) -> RunResult:
    """Run a prepared ``runner`` on every read of ``suite`` and record the run in the layout.

    The reads are ``layout.reads(suite)``; the predictions, logs and run record are the layout's
    ``predictions``, ``logs_dir`` and ``run_json`` of ``(runner.name, suite, arm)``. The provenance gains the
    reads' path and SHA-256 (``reads``) and, when given, ``settings``: the tool settings the run was made with,
    which an orchestrator compares before it reuses the run. ``run.json`` and the old predictions are removed
    first and ``run.json`` is written last, so a run that raises leaves no record behind; a run that returns a
    failure is recorded with its return code.
    """
    run_json = layout.run_json(runner.name, suite, arm)
    run_json.unlink(missing_ok=True)
    predictions = layout.predictions(runner.name, suite, arm)
    predictions.unlink(missing_ok=True)
    predictions.parent.mkdir(parents=True, exist_ok=True)
    reads = layout.reads(suite)
    result = runner.run(reads, predictions, suite=suite, arm=arm, log_dir=layout.logs_dir(runner.name, suite, arm),
                        alleles=alleles)
    provenance = {**result.provenance, "reads": {"path": str(reads), "sha256": sha256_file(reads)}}
    if settings is not None:
        provenance["settings"] = settings
    result = dataclasses.replace(result, provenance=provenance)
    write_run_json(result, run_json)
    return result


def _peak_rss_kb(path: Path) -> int | None:
    """The last line GNU time wrote (it may first note a non-zero exit status), as an integer; None otherwise."""
    try:
        words = path.read_text(encoding="utf-8").split()
    except OSError:
        return None
    return int(words[-1]) if words and words[-1].isdigit() else None
