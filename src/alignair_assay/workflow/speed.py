"""The speed protocol: each tool's cold start, steady throughput and peak memory, on one host and the same reads.

A configuration is a tool at a thread count, a CPU list and a device (``speed_configs``). It annotates two read
sets sampled evenly across S1 (``speed_set``), so both have S1's mix of read designs and SHM levels: a tiny one of
``n_tiny`` reads and a full one of ``n_full`` reads, ``repeats``
times each, alternating (``measure``). Every run is a fresh process (a fresh container for the AlignAIR image), so
a tiny run's wall time is the cold start: starting, loading the model or the germline databases, and ``n_tiny``
reads. The full median minus the tiny median is the time of ``n_full - n_tiny`` more reads, which gives the steady
throughput without the start-up cost. ``run_speed`` measures every configuration and writes ``speed/speed.json``:
the host (``host_info``), the protocol and, per configuration, its settings and measurements. Wall times are in
seconds and peak resident memory in KB.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import statistics
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from alignair_assay.core.cases import read_fasta, write_fasta
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import sha256_file
from alignair_assay.core.reference import Reference
from alignair_assay.runners import make_runner
from alignair_assay.runners.alignair import LOCAL_CODE_IDENTITY, local_code_identity, local_cuda_available
from alignair_assay.runners.base import Runner, RunnerOptions
from alignair_assay.suites import get_suite
from alignair_assay.suites.engine import requested_count
from alignair_assay.workflow.summary import ORACLE, read_json

if TYPE_CHECKING:
    from alignair_assay.workflow.orchestrate import RunConfig

SPEED_KIND = "alignair_assay.speed"
SPEED_VERSION = "1"
# The suite whose reads are timed, and the arm name the timed runs record.
SPEED_SUITE = "S1"
SPEED_ARM = "speed"
# Protocol defaults: reads in the tiny and the full set, and runs of each.
N_TINY = 50
N_FULL = 4000
REPEATS = 3
# The tool timed on CUDA as well as on CPU.
LOCAL = "alignair-local"
# What each result records about its configuration; a kept result must match on all of them. alignair-local also
# records the checkpoint's digest and the code its CLI runs (``runners.alignair.local_code_identity``).
IDENTITY = ("tool", "threads", "cpus", "device", "checkpoint_sha256", *LOCAL_CODE_IDENTITY)
# Seconds to wait for nvidia-smi or the docker daemon before recording them as unavailable.
_PROBE_TIMEOUT = 20


@dataclass(frozen=True)
class SpeedConfig:
    """One timed configuration: ``tool`` built from ``options``, whose ``threads``, ``cpus`` and ``device`` are the
    configuration's. Calling it builds the runner, unprepared (``runners.make_runner``)."""

    tool: str
    options: RunnerOptions

    def __call__(self) -> Runner:
        return make_runner(self.tool, self.options)

    def identity(self) -> dict:
        """The ``tool``, ``threads``, ``cpus`` (the CPU list it is pinned to) and torch ``device`` (``cpu`` for every
        tool but alignair-local), and for alignair-local the ``checkpoint_sha256`` (None for a missing file) and the
        code its CLI runs (``runners.alignair.local_code_identity``: ``code_sha256``, ``git_commit``,
        ``git_dirty``, ``torch``)."""
        record = {"tool": self.tool, "threads": self.options.threads, "cpus": self.options.cpus,
                  "device": self.options.device if self.tool == LOCAL else "cpu"}
        if self.tool == LOCAL:
            checkpoint = self.options.alignair_checkpoint
            record["checkpoint_sha256"] = sha256_file(checkpoint) if checkpoint and Path(checkpoint).is_file() else None
            record.update(local_code_identity())
        return record


def cpu_list(threads: int) -> str:
    """The CPU list of a ``threads``-thread configuration: CPUs 0 to threads - 1 (``"0"``, ``"0-7"``)."""
    return "0" if threads == 1 else f"0-{threads - 1}"


def speed_configs(cfg: RunConfig, *, cuda: bool | None = None) -> list[tuple[str, SpeedConfig]]:
    """The configurations timed for ``cfg``, by name, in order. Each tool of ``cfg.tools`` but the oracle runs at
    1 thread pinned to CPU 0 (``<tool>_1t``) and at N = ``cfg.threads`` threads pinned to CPUs 0 to N - 1
    (``<tool>_<N>t``; one configuration when N is 1). alignair-local, timed when it is a tool or
    ``cfg.alignair_checkpoint`` is set, runs instead on ``cuda`` with N threads (``alignair-local_cuda``) when
    its interpreter sees a CUDA device (``cuda``; probed by ``runners.alignair.local_cuda_available`` when None),
    and on ``cpu`` at N threads. The tools' other options are ``cfg``'s. ValueError when N is below 1 or above the
    host's logical CPUs."""
    n = cfg.threads
    logical = os.cpu_count() or 1
    if n < 1 or n > logical:
        raise ValueError(f"the speed protocol pins N = {n} threads to CPUs 0 to {n - 1}; N must be from 1 to the "
                         f"host's {logical} logical CPUs")
    tools = [tool for tool in cfg.tools if tool != ORACLE]
    if cfg.alignair_checkpoint and LOCAL not in tools:
        tools.append(LOCAL)
    base = RunnerOptions(igblast_dir=cfg.igblast_dir, alignair_image=cfg.alignair_image,
                         alignair_model=cfg.alignair_model, alignair_checkpoint=cfg.alignair_checkpoint)
    configs = []
    for tool in tools:
        if tool == LOCAL:
            if cuda is None:
                cuda = local_cuda_available()
            on_gpu = [("cuda", f"{tool}_cuda")] if cuda else []
            for device, name in [*on_gpu, ("cpu", f"{tool}_{n}t")]:
                configs.append((name, SpeedConfig(tool, replace(base, threads=n, cpus=cpu_list(n), device=device))))
        else:
            for threads in dict.fromkeys((1, n)):
                configs.append((f"{tool}_{threads}t",
                                SpeedConfig(tool, replace(base, threads=threads, cpus=cpu_list(threads)))))
    return configs


def check_speed(cfg: RunConfig) -> None:
    """ValueError when ``all`` cannot time ``cfg``: S1 not among its suites, S1 at ``cfg.scale`` with fewer than
    ``N_FULL`` reads, a thread count ``speed_configs`` refuses, or a configuration whose runner cannot be built
    (its tool's options); FileNotFoundError for an ``alignair_checkpoint`` that is not a file. Nothing is run."""
    if cfg.alignair_checkpoint and not Path(cfg.alignair_checkpoint).expanduser().is_file():
        raise FileNotFoundError(f"no AlignAIR checkpoint {cfg.alignair_checkpoint} to time")
    if SPEED_SUITE not in cfg.suites:
        raise ValueError(f"--speed times the reads of {SPEED_SUITE}: add {SPEED_SUITE} to --suites")
    strata = get_suite(SPEED_SUITE).strata
    reads = sum(requested_count(stratum, cfg.scale) for stratum in strata)
    if reads < N_FULL:
        minimum = N_FULL / sum(stratum.n for stratum in strata)
        raise ValueError(f"--speed times {N_FULL} reads of {SPEED_SUITE}, which has {reads} at scale "
                         f"{cfg.scale}; use a scale of at least {minimum:g}")
    for _, config in speed_configs(cfg, cuda=False):
        config()


def speed_set(layout: RunLayout, n: int) -> Path:
    """Write ``layout.speed_reads(n)``, ``n`` reads evenly spaced across S1's ``reads.fasta``, and return its path.

    Of S1's N reads, the set holds the reads at positions ``floor(i * N / n)`` for i from 0 to n - 1, in file order
    (every (N / n)-th read when that is whole). S1 lists its reads stratum by stratum, so the first n reads would
    over-represent its first read designs; an evenly spaced sample holds every stratum in about its share of S1. The
    same S1 gives the same file. FileNotFoundError when S1 is not built; ValueError when ``n`` is below 1 or S1 has
    fewer than ``n`` reads."""
    path = layout.speed_reads(n)
    source = layout.reads(SPEED_SUITE)
    if not source.is_file():
        raise FileNotFoundError(f"no {source}: build {SPEED_SUITE} first, the speed protocol times its reads")
    records = read_fasta(source)
    if len(records) < n:
        raise ValueError(f"the speed protocol needs {n} reads of {SPEED_SUITE}, but {source} has {len(records)}; "
                         "build it at a larger scale or time fewer reads")
    sample = [records[index * len(records) // n] for index in range(n)]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_fasta(({"case_id": name, "sequence": sequence} for name, sequence in sample), path)
    return path


def measure(make_runner: Callable[[], Runner], fasta_tiny, fasta_full, n_tiny: int, n_full: int, *,
            repeats: int = REPEATS, work_dir=None) -> dict:
    """Time one runner on a tiny and a full read set, alternating tiny then full, ``repeats`` times each.

    ``make_runner()`` is called once and returns the runner, ready to run (prepared). Each run annotates
    ``fasta_tiny`` (``n_tiny`` reads) or ``fasta_full`` (``n_full`` reads) as suite ``SPEED_SUITE``, arm
    ``SPEED_ARM``, into ``work_dir/tiny`` or ``work_dir/full`` (predictions and logs; each run replaces the last
    one of its size, and a temporary directory, removed afterwards, stands in for a None ``work_dir``). Returns:

    - ``cold_start_s``: the median wall time of the tiny runs;
    - ``steady_reads_per_s``: ``(n_full - n_tiny) / (median_full_s - median_tiny_s)``, None when the full median
      is not above the tiny one;
    - ``median_tiny_s`` and ``median_full_s``;
    - ``peak_rss_kb``: the largest peak over all runs, None when no run measured one;
    - ``walls``: ``{"tiny": [...], "full": [...]}``, every run's wall time in run order;
    - ``command``: the last full run's command.

    RuntimeError naming the size and repeat when a run fails (a non-zero return code, or no predictions), with
    the tool's error line; ValueError when a run reports another read count than its set's, or when ``n_tiny`` is
    below 1, ``n_full`` not above it or ``repeats`` below 1.
    """
    if n_tiny < 1 or n_full <= n_tiny or repeats < 1:
        raise ValueError(f"the speed protocol needs 1 <= n_tiny < n_full and at least one repeat, got n_tiny "
                         f"{n_tiny}, n_full {n_full}, repeats {repeats}")
    walls: dict[str, list[float]] = {"tiny": [], "full": []}
    peaks: list[int] = []
    command = None
    scratch = tempfile.TemporaryDirectory(prefix="alignair_assay_speed_") if work_dir is None else None
    with scratch or nullcontext():
        root = Path(scratch.name if scratch else work_dir)
        runner = make_runner()
        for repeat in range(1, repeats + 1):
            for size, fasta, n in (("tiny", fasta_tiny, n_tiny), ("full", fasta_full, n_full)):
                directory = root / size
                shutil.rmtree(directory, ignore_errors=True)
                directory.mkdir(parents=True)
                out_tsv = directory / "predictions.tsv"
                result = runner.run(Path(fasta), out_tsv, suite=SPEED_SUITE, arm=SPEED_ARM, log_dir=directory / "logs")
                if result.returncode != 0 or not out_tsv.is_file():
                    reason = result.provenance.get("error") or "no predictions were written"
                    raise RuntimeError(f"{runner.name}: the {size} run {repeat} of {repeats} failed with return code "
                                       f"{result.returncode}: {reason}")
                if result.n_reads != n:
                    raise ValueError(f"{runner.name}: the {size} run read {result.n_reads} reads, expected {n}")
                walls[size].append(result.wall_seconds)
                if result.peak_rss_kb is not None:
                    peaks.append(result.peak_rss_kb)
                if size == "full":
                    command = result.commands[-1] if result.commands else None
    tiny, full = statistics.median(walls["tiny"]), statistics.median(walls["full"])
    return {"cold_start_s": tiny, "steady_reads_per_s": (n_full - n_tiny) / (full - tiny) if full > tiny else None,
            "median_tiny_s": tiny, "median_full_s": full, "peak_rss_kb": max(peaks) if peaks else None,
            "walls": walls, "command": command}


def host_info() -> dict:
    """The measuring host: ``cpu`` (the CPU model name; None when unknown), ``logical_cpus``, ``ram_gb`` (total
    memory in GiB, one decimal; None when unknown), ``gpu`` (the NVIDIA GPU names nvidia-smi lists, comma-joined;
    None without one), ``docker`` (the docker daemon's version; None without a reachable daemon) and ``kernel``
    (its release)."""
    return {"cpu": _cpu_model(), "logical_cpus": os.cpu_count(), "ram_gb": _ram_gb(),
            "gpu": ", ".join(_command_lines(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])) or None,
            "docker": "".join(_command_lines(["docker", "version", "--format", "{{.Server.Version}}"])) or None,
            "kernel": platform.release()}


def run_speed(cfg: RunConfig, layout: RunLayout, reference: Reference, *, n_tiny: int | None = None,
              n_full: int | None = None, repeats: int | None = None, names: Iterable[str] | None = None,
              config_hash: str | None = None, reuse: bool = False) -> dict:
    """Time the configurations of ``cfg`` (``speed_configs``; only ``names`` when given) on S1 of ``layout`` and
    write ``layout.speed_json``; returns what it holds.

    ``n_tiny``, ``n_full`` and ``repeats`` default to ``N_TINY``, ``N_FULL`` and ``REPEATS``. Every configuration's
    runner is built before anything is measured, so a missing tool option, an unknown name or an empty selection
    raises ValueError first. The read sets are written (``speed_set``), then each configuration is prepared
    against ``reference``, fingerprinted and measured (``measure``, in ``layout.speed_work_dir(name)``), and
    ``speed.json`` is rewritten after each one. A configuration that fails (to prepare or to run) is recorded with
    its ``error``, and the others go on.

    ``speed.json`` holds ``kind``, ``version``, the ``host`` (``host_info``), ``n_tiny``, ``n_full``, ``repeats``,
    the timed ``reads`` (suite and SHA-256 of its ``reads.fasta``), the ``config_hash`` of the ``all``
    configuration it belongs to (None outside ``all``) and ``results``: per configuration, its identity
    (``SpeedConfig.identity``), ``measured_utc`` and either its ``settings`` (``Runner.fingerprint``) and
    ``measure``'s values, or its ``error``. With ``reuse``, a result of the existing ``speed.json`` is kept, not
    measured again, when the file has the same host, protocol, reads and config hash and the result succeeded with
    the same identity.
    """
    n_tiny, n_full = N_TINY if n_tiny is None else n_tiny, N_FULL if n_full is None else n_full
    repeats = REPEATS if repeats is None else repeats
    if n_tiny < 1 or n_full <= n_tiny or repeats < 1:
        raise ValueError(f"the speed protocol needs 1 <= n_tiny < n_full and at least one repeat, got n_tiny "
                         f"{n_tiny}, n_full {n_full}, repeats {repeats}")
    configs = speed_configs(cfg)
    if names is not None:
        wanted = list(dict.fromkeys(names))
        known = dict(configs)
        unknown = [name for name in wanted if name not in known]
        if unknown:
            raise ValueError(f"no speed configuration {unknown[0]!r}; the configurations are {', '.join(known)}")
        configs = [(name, known[name]) for name in wanted]
    if not configs:
        raise ValueError(f"no tool to time among {', '.join(cfg.tools)} (the oracle is not timed)")
    for _, config in configs:
        config()
    tiny, full = speed_set(layout, n_tiny), speed_set(layout, n_full)
    protocol = {"host": host_info(), "n_tiny": n_tiny, "n_full": n_full, "repeats": repeats,
                "reads": {"suite": SPEED_SUITE, "sha256": sha256_file(layout.reads(SPEED_SUITE))},
                "config_hash": config_hash}
    previous = (read_json(layout.speed_json) or {}) if reuse else {}
    same = all(previous.get(key) == value for key, value in protocol.items())
    kept = (previous.get("results") or {}) if same else {}
    record = {"kind": SPEED_KIND, "version": SPEED_VERSION, **protocol, "results": {}}
    for name, config in configs:
        identity = config.identity()
        old = kept.get(name)
        if old and "error" not in old and all(old.get(key) == identity.get(key) for key in IDENTITY):
            record["results"][name] = old
            print(f"speed {name}: up to date, skipped", flush=True)
            continue
        entry = {**identity, "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        try:
            runner = config()
            runner.prepare(layout, reference)
            entry["settings"] = runner.fingerprint()
            entry.update(measure(lambda: runner, tiny, full, n_tiny, n_full, repeats=repeats,
                                 work_dir=layout.speed_work_dir(name)))
        except (RuntimeError, ValueError, OSError) as error:
            entry = {**identity, "measured_utc": entry["measured_utc"], "error": f"{type(error).__name__}: {error}"}
        record["results"][name] = entry
        print(_speed_line(name, entry), flush=True)
        # Written after every configuration, so an interrupted protocol keeps what it measured for a resume.
        _write(layout, record)
    _write(layout, record)
    return record


def _write(layout: RunLayout, record: dict) -> None:
    layout.speed_dir.mkdir(parents=True, exist_ok=True)
    layout.speed_json.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _speed_line(name: str, entry: dict) -> str:
    if "error" in entry:
        return f"speed {name}: failed: {entry['error']}"
    steady = entry["steady_reads_per_s"]
    peak = entry["peak_rss_kb"]
    return (f"speed {name}: cold start {entry['cold_start_s']:.2f} s, steady "
            f"{'not measurable' if steady is None else f'{steady:.1f} reads/s'}, median full run "
            f"{entry['median_full_s']:.2f} s, peak RSS {'not measured' if peak is None else f'{peak / 1024:.1f} MB'}")


def _cpu_model() -> str | None:
    """The first ``model name`` of /proc/cpuinfo, else ``platform.processor()``; None when neither says."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "model name" and value.strip():
                return value.strip()
    except OSError:
        pass
    return platform.processor() or None


def _ram_gb() -> float | None:
    """Total memory in GiB from /proc/meminfo (``MemTotal``, in KiB), one decimal; None when unreadable."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 ** 2, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _command_lines(argv: list[str]) -> list[str]:
    """The non-empty stripped stdout lines of ``argv``; empty when it is missing, fails or times out."""
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=_PROBE_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]
