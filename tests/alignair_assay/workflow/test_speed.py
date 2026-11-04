"""The speed protocol: the timed read sets, the measurement (cold start, steady throughput, peak RSS), the timed
configurations, the host record, ``speed.json`` under ``all --speed`` and the ``speed`` command, and the report's
speed section."""
import dataclasses
import json
import os
import platform
import shutil
import statistics
import time
from pathlib import Path

import pytest

pytest.importorskip("GenAIRR")

from alignair_assay.cli import main  # noqa: E402
from alignair_assay.core.cases import read_fasta  # noqa: E402
from alignair_assay.core.layout import RunLayout  # noqa: E402
from alignair_assay.core.provenance import sha256_file, tree_digest  # noqa: E402
from alignair_assay.core.reference import DEFAULT_REFERENCE, Reference  # noqa: E402
from alignair_assay.report.run import run_report_markdown  # noqa: E402
from alignair_assay.runners import RUNNERS  # noqa: E402
from alignair_assay.runners import alignair as alignair_module  # noqa: E402
from alignair_assay.runners.alignair import AlignAIRImage, AlignAIRLocal, local_code_identity  # noqa: E402
from alignair_assay.runners.base import RunResult, fasta_ids  # noqa: E402
from alignair_assay.runners.igblast import IgBlastRunner  # noqa: E402
from alignair_assay.runners.oracle import OracleRunner  # noqa: E402
from alignair_assay.suites.engine import DEFAULT_SEED  # noqa: E402
from alignair_assay.workflow import speed  # noqa: E402
from alignair_assay.workflow.build import build_suite  # noqa: E402
from alignair_assay.workflow.orchestrate import RunConfig, check_config, run_all  # noqa: E402
from alignair_assay.workflow.speed import host_info, measure, speed_configs, speed_set  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.genairr_build]  # these tests build GenAIRR suites
HOST_KEYS = {"cpu", "logical_cpus", "ram_gb", "gpu", "docker", "kernel"}


class Sleeper:
    """A runner whose run sleeps ``start`` seconds plus ``per_read`` seconds per read and reports a peak RSS of
    ``1000 + reads`` KB (None with ``rss=False``); it records the FASTA, suite and arm of every run."""

    name = "sleeper"

    def __init__(self, start=0.2, per_read=0.001, rss=True, fail=None, reads=None):
        self.start, self.per_read, self.rss, self.fail, self.reads = start, per_read, rss, fail, reads
        self.calls = []

    def fingerprint(self):
        return {}

    def prepare(self, layout, reference):
        return {}

    def run(self, fasta, out_tsv, *, suite, arm, log_dir, alleles=None):
        n = len(fasta_ids(fasta))
        self.calls.append((Path(fasta).name, suite, arm))
        began = time.perf_counter()
        time.sleep(self.start + self.per_read * n)
        failed = self.fail is not None and len(self.calls) == self.fail
        if not failed:
            Path(out_tsv).write_text("sequence_id\n")
        return RunResult(tool=self.name, suite=suite, arm=arm, predictions=str(out_tsv),
                         n_reads=self.reads or n, wall_seconds=time.perf_counter() - began,
                         peak_rss_kb=1000 + n if self.rss else None, returncode=1 if failed else 0,
                         commands=[["sleep", str(n)]], provenance={"error": "Killed"} if failed else {})


class Quick(OracleRunner):
    """Oracle rows under the name ``quick``, with a peak RSS and a command: the tool the speed runs time."""

    name = "quick"

    def __init__(self, options):
        super().__init__()
        self.options = options

    def fingerprint(self):
        return {"version": "1"}

    def run(self, fasta, out_tsv, *, suite, arm, log_dir, alleles=None):
        result = super().run(fasta, out_tsv, suite=suite, arm=arm, log_dir=log_dir)
        return dataclasses.replace(result, tool=self.name, peak_rss_kb=2048, commands=[["quick", str(fasta)]])


class Brittle(Quick):
    """``quick`` that fails every run on one thread, as a tool killed on a small CPU budget would."""

    name = "brittle"

    def run(self, fasta, out_tsv, *, suite, arm, log_dir, alleles=None):
        if arm == speed.SPEED_ARM and self.options.threads == 1:
            return RunResult(tool=self.name, suite=suite, arm=arm, predictions=str(out_tsv),
                             n_reads=len(fasta_ids(fasta)), wall_seconds=0.0, peak_rss_kb=None, returncode=137,
                             commands=[], provenance={"error": "Killed"})
        return super().run(fasta, out_tsv, suite=suite, arm=arm, log_dir=log_dir)


@pytest.fixture(autouse=True)
def fake_tools(monkeypatch):
    monkeypatch.setitem(RUNNERS, "quick", Quick)
    monkeypatch.setitem(RUNNERS, "brittle", Brittle)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A run directory with S1 built at scale 0.02 (100 reads), as ``all`` builds it."""
    root = tmp_path_factory.mktemp("s1")
    build_suite("S1", RunLayout(root), reference_name=DEFAULT_REFERENCE, split="dev", seed=DEFAULT_SEED, scale=0.02,
                workers=2)
    return root


def _copy(root: Path, tmp_path: Path) -> Path:
    target = tmp_path / "run"
    shutil.copytree(root, target)
    return target


def _fasta(path: Path, n: int) -> Path:
    path.write_text("".join(f">S1:read:{index:06d}\n{'ACGT' * 10}\n" for index in range(n)))
    return path


def _s1_reads(layout: RunLayout, n: int) -> list[tuple[str, str]]:
    """Write ``n`` reads as S1's presented reads and return them."""
    layout.suite_dir("S1").mkdir(parents=True)
    _fasta(layout.reads("S1"), n)
    return read_fasta(layout.reads("S1"))


@pytest.fixture
def small_protocol(monkeypatch):
    """The protocol at 10 and 60 reads, 2 runs each, so S1 at scale 0.02 can be timed."""
    monkeypatch.setattr(speed, "N_TINY", 10)
    monkeypatch.setattr(speed, "N_FULL", 60)
    monkeypatch.setattr(speed, "REPEATS", 2)


def test_measure_recovers_linear_cost(tmp_path):
    tiny, full = _fasta(tmp_path / "tiny.fasta", 50), _fasta(tmp_path / "full.fasta", 500)
    runner = Sleeper()
    built = []
    result = measure(lambda: built.append(runner) or runner, tiny, full, 50, 500, repeats=3,
                     work_dir=tmp_path / "work")
    assert built == [runner]  # the runner is made once
    # Runs alternate tiny, full; each is a run of S1's reads in the speed arm.
    assert runner.calls == [("tiny.fasta", "S1", "speed"), ("full.fasta", "S1", "speed")] * 3
    # 0.2 s + 1 ms per read: the cold start is the tiny run (0.25 s), the steady rate 1000 reads/s.
    assert result["cold_start_s"] == pytest.approx(0.25, rel=0.15)
    assert result["steady_reads_per_s"] == pytest.approx(1000, rel=0.15)
    assert result["median_full_s"] == pytest.approx(0.7, rel=0.15)
    assert result["cold_start_s"] == result["median_tiny_s"] == statistics.median(result["walls"]["tiny"])
    assert result["median_full_s"] == statistics.median(result["walls"]["full"])
    assert len(result["walls"]["tiny"]) == len(result["walls"]["full"]) == 3
    assert result["peak_rss_kb"] == 1500 and result["command"] == ["sleep", "500"]
    # The last run of each size is kept in the work directory.
    assert (tmp_path / "work" / "full" / "predictions.tsv").is_file()
    assert (tmp_path / "work" / "tiny" / "predictions.tsv").is_file()


def test_measure_edge_cases(tmp_path):
    tiny, full = _fasta(tmp_path / "tiny.fasta", 5), _fasta(tmp_path / "full.fasta", 20)
    instant = Sleeper(start=0.0, per_read=0.0, rss=False)
    result = measure(lambda: instant, tiny, full, 5, 20, repeats=1)  # a temporary work directory
    assert result["peak_rss_kb"] is None and len(result["walls"]["full"]) == 1
    # A failed run stops the measurement, naming the run and the tool's error.
    with pytest.raises(RuntimeError, match=r"full run 2 of 2 failed with return code 1: Killed"):
        measure(lambda: Sleeper(start=0.0, per_read=0.0, fail=4), tiny, full, 5, 20, repeats=2)
    # A run that reads another number of reads than its set holds would give a wrong rate.
    with pytest.raises(ValueError, match="read 7 reads, expected 5"):
        measure(lambda: Sleeper(start=0.0, per_read=0.0, reads=7), tiny, full, 5, 20, repeats=1)
    for n_tiny, n_full, repeats in ((0, 20, 1), (20, 20, 1), (5, 20, 0)):
        with pytest.raises(ValueError, match="n_tiny < n_full"):
            measure(lambda: instant, tiny, full, n_tiny, n_full, repeats=repeats)


def test_speed_set_deterministic(tmp_path):
    layout = RunLayout(tmp_path)
    records = _s1_reads(layout, 10)
    path = speed_set(layout, 4)
    assert path == layout.speed_reads(4) == tmp_path / "speed" / "reads_4.fasta"
    # Evenly spaced over the 10 reads, in file order: reads floor(i * 10 / 4) for i in 0..3.
    assert read_fasta(path) == [records[0], records[2], records[5], records[7]]
    first = path.read_bytes()
    assert speed_set(layout, 4).read_bytes() == first
    assert read_fasta(speed_set(layout, 10)) == records
    # Every second read when N / n is whole.
    assert read_fasta(speed_set(layout, 5)) == records[::2]


def test_speed_set_reflects_the_design_mix(tmp_path):
    # S1's composition at scale 1.0, stratum by stratum as its reads.fasta lists them: 4 designs x (igm 500,
    # igg 500, high 250).
    layout = RunLayout(tmp_path)
    layout.suite_dir("S1").mkdir(parents=True)
    strata = [(f"{design}_{shm}", count) for design in ("full", "fr1", "fr2", "fr3")
              for shm, count in (("igm", 500), ("igg", 500), ("high", 250))]
    layout.reads("S1").write_text("".join(f">S1:{stratum}:{index:06d}\nACGT\n" for stratum, count in strata
                                          for index in range(count)))
    total = sum(count for _, count in strata)
    for n in (50, 1000, 4000):
        names = [name for name, _ in read_fasta(speed_set(layout, n))]
        assert len(names) == len(set(names)) == n
        shares = {stratum: sum(name.split(":")[1] == stratum for name in names) / n for stratum, _ in strata}
        for stratum, count in strata:
            assert abs(shares[stratum] - count / total) <= 0.05, (n, stratum, shares[stratum])
        # Each design is a quarter of the set, within 5 points (the first 4000 reads would hold 6% FR3).
        for design in ("full", "fr1", "fr2", "fr3"):
            share = sum(value for stratum, value in shares.items() if stratum.startswith(design))
            assert abs(share - 0.25) <= 0.05, (n, design, share)


def test_speed_set_too_small_raises(tmp_path):
    layout = RunLayout(tmp_path)
    with pytest.raises(FileNotFoundError, match="build S1"):
        speed_set(layout, 4)
    _s1_reads(layout, 10)
    with pytest.raises(ValueError, match="needs 11 reads of S1, but .* has 10"):
        speed_set(layout, 11)
    assert not layout.speed_reads(11).exists()
    for bad in (0, -1, True, 2.5):
        with pytest.raises(ValueError):
            layout.speed_reads(bad)


def test_host_info_keys(tmp_path, monkeypatch):
    info = host_info()
    assert set(info) == HOST_KEYS
    assert info["logical_cpus"] == os.cpu_count() and info["kernel"] == platform.release()
    assert isinstance(info["ram_gb"], float) and info["ram_gb"] > 0
    assert info["cpu"] is None or isinstance(info["cpu"], str)
    json.dumps(info, allow_nan=False)
    # Without nvidia-smi or docker, the host has no GPU and no docker daemon to report: None, not an error.
    monkeypatch.setenv("PATH", str(tmp_path))
    bare = host_info()
    assert bare["gpu"] is None and bare["docker"] is None and bare["logical_cpus"] == os.cpu_count()


def test_speed_configs():
    cfg = RunConfig(suites=("S1",), tools=("igblast", "alignair"), threads=8, igblast_dir="/opt/igblast",
                    alignair_checkpoint="/models/m.alignair")
    configs = dict(speed_configs(cfg, cuda=True))
    assert list(configs) == ["igblast_1t", "igblast_8t", "alignair_1t", "alignair_8t", "alignair-local_cuda",
                             "alignair-local_8t"]
    identities = {name: config.identity() for name, config in configs.items()}
    assert identities["igblast_1t"] == {"tool": "igblast", "threads": 1, "cpus": "0", "device": "cpu"}
    assert identities["alignair_8t"] == {"tool": "alignair", "threads": 8, "cpus": "0-7", "device": "cpu"}
    # A missing checkpoint has no digest; the code the local CLI runs is part of its identity.
    assert identities["alignair-local_cuda"] == {"tool": "alignair-local", "threads": 8, "cpus": "0-7",
                                                 "device": "cuda", "checkpoint_sha256": None,
                                                 **local_code_identity()}
    assert set(speed.IDENTITY) >= {"checkpoint_sha256", "code_sha256", "torch_version"}
    assert not {"git_commit", "git_dirty"} & set(speed.IDENTITY)
    assert identities["alignair-local_8t"]["device"] == "cpu"
    # Each builds its tool's runner with the configuration's threads, CPUs and device, and the run's options.
    igblast = configs["igblast_1t"]()
    assert isinstance(igblast, IgBlastRunner) and (igblast.threads, igblast.cpus) == (1, "0")
    assert igblast.igblast_dir == Path("/opt/igblast")
    image = configs["alignair_8t"]()
    assert isinstance(image, AlignAIRImage) and (image.threads, image.cpus) == (8, "0-7")
    local = configs["alignair-local_cuda"]()
    assert isinstance(local, AlignAIRLocal) and (local.device, local.threads, local.cpus) == ("cuda", 8, "0-7")
    assert local.checkpoint == Path("/models/m.alignair")
    # the CLI times at its own default batch size, as a user runs it
    assert local.batch_size is None and configs["alignair-local_8t"]().batch_size is None

    # No CUDA device: alignair-local on CPU only. No checkpoint: no alignair-local. The oracle is never timed,
    # and one thread is one configuration.
    assert [name for name, _ in speed_configs(cfg, cuda=False)][4:] == ["alignair-local_8t"]
    single = dataclasses.replace(cfg, tools=("oracle", "igblast"), threads=1, alignair_checkpoint=None)
    assert [name for name, _ in speed_configs(single)] == ["igblast_1t"]
    # alignair-local as a tool is timed once, the same way.
    local_tool = dataclasses.replace(cfg, tools=("alignair-local", "igblast"))
    assert [name for name, _ in speed_configs(local_tool, cuda=False)] == [
        "alignair-local_8t", "igblast_1t", "igblast_8t"]
    for threads in (0, (os.cpu_count() or 1) + 1):
        with pytest.raises(ValueError, match="logical CPUs"):
            speed_configs(dataclasses.replace(cfg, threads=threads), cuda=False)


def test_speed_refuses_alignair_api(built, tmp_path, capsys):
    # The speed protocol does not know alignair-api's checkpoint identity: alignair-local times the same checkpoint
    # and product code. Refused by the configurations, by all --speed and by the speed command, before any timing.
    cfg = RunConfig(suites=("S1",), tools=("igblast", "alignair-api"), threads=2, igblast_dir="/opt/igblast",
                    alignair_checkpoint="/models/m.alignair", scale=0.8, speed=True)
    with pytest.raises(ValueError, match="alignair-api.*alignair-local"):
        speed_configs(cfg, cuda=False)
    with pytest.raises(ValueError, match="alignair-api"):
        check_config(cfg)
    root = _copy(built, tmp_path)
    assert main(["speed", "--run-dir", str(root), "--tools", "alignair-api", "--alignair-checkpoint",
                 "/models/m.alignair", "--threads", "2"]) == 2
    assert "alignair-api" in capsys.readouterr().err
    assert not RunLayout(root).speed_json.exists()


def test_speed_checked_with_the_config():
    base = dict(suites=("S0", "S1"), tools=("oracle", "quick"), scale=0.8, speed=True)
    check_config(RunConfig(**base))
    for overrides, message in ((dict(suites=("S0",)), "add S1 to --suites"),
                               (dict(scale=0.02), r"S1, which has 100 at scale 0.02; use a scale of at least 0.8"),
                               (dict(threads=(os.cpu_count() or 1) + 1), "logical CPUs")):
        with pytest.raises(ValueError, match=message):
            check_config(RunConfig(**(base | overrides)))
    # A checkpoint timed as alignair-local must exist.
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        check_config(RunConfig(**(base | dict(alignair_checkpoint="/nowhere/m.alignair"))))
    # Without --speed, none of this is asked.
    check_config(RunConfig(**(base | dict(suites=("S0",), scale=0.02, speed=False))))


def _speed_record(**overrides) -> dict:
    host = {"cpu": "Test CPU", "logical_cpus": 16, "ram_gb": 125.7, "gpu": "Test GPU", "docker": "29.9.1",
            "kernel": "7.0.0-test"}
    results = {
        "igblast_1t": {"tool": "igblast", "threads": 1, "cpus": "0", "device": "cpu", "cold_start_s": 0.52,
                       "steady_reads_per_s": 41.25, "median_tiny_s": 0.52, "median_full_s": 96.27,
                       "peak_rss_kb": 204800, "walls": {"tiny": [0.52], "full": [96.27]}},
        "alignair_1t": {"tool": "alignair", "threads": 1, "cpus": "0", "device": "cpu",
                        "error": "RuntimeError: alignair: the full run 1 of 3 failed with return code 137: Killed"},
        "alignair-local_cuda": {"tool": "alignair-local", "threads": 8, "cpus": "0-7", "device": "cuda",
                                "checkpoint_sha256": "ab" * 32, "cold_start_s": 9.5, "steady_reads_per_s": None,
                                "median_tiny_s": 9.5, "median_full_s": 9.4, "peak_rss_kb": None,
                                "walls": {"tiny": [9.5], "full": [9.4]}},
    }
    return {"kind": "alignair_assay.speed", "version": "2", "host": host, "n_tiny": 50, "n_full": 4000,
            "repeats": 3, "reads": {"suite": "S1", "sha256": "0" * 64}, "config_hash": "c" * 64, "results": results,
            "failed": [{"name": "alignair_1t", "error": results["alignair_1t"]["error"]}]} | overrides


def _summary(speed_record) -> dict:
    return {"a": "igblast", "b": "alignair", "config_hash": "c" * 64, "suites": [], "speed": speed_record,
            "config": {"split": "dev", "seed": 1, "scale": 1.0, "reference": "HUMAN_IGH_OGRDB"}}


def test_speed_section_rendered():
    md = run_report_markdown(_summary(_speed_record()))
    section = md[md.index("## Speed"):]
    assert ("Host: Test CPU, 16 logical CPUs, 125.7 GB RAM, GPU Test GPU, Docker 29.9.1, kernel 7.0.0-test."
            in section)
    assert "50 and 4000 reads evenly spaced across S1, 3 times each" in section
    assert "container start" in section and "memory.peak" in section and "GNU time" in section
    assert ("| Configuration | Cold start (s) | Steady reads/s | Median full run (s) | Peak RSS (MB) | Threads | CPUs "
            "| Device |") in section
    assert "| igblast_1t | 0.52 | 41.2 | 96.27 | 200.0 | 1 | 0 | cpu |" in section
    # A steady rate that could not be measured, and a peak that was not, show dashes.
    assert "| alignair-local_cuda | 9.50 | — | 9.40 | — | 8 | 0-7 | cuda |" in section
    # A failed configuration has no values; its error is listed under the table.
    assert "| alignair_1t | — | — | — | — | 1 | 0 | cpu |" in section
    assert "- alignair_1t failed: RuntimeError: alignair: the full run 1 of 3 failed with return code 137: Killed" \
        in section


def test_stale_speed_named_not_shown():
    md = run_report_markdown(_summary({"stale": True, "config_hash": "d" * 64}))
    section = md[md.index("## Speed"):]
    assert "another configuration" in section and "dddddddddddd" in section and "all --speed" in section
    assert "| Configuration |" not in section
    # One the speed command wrote has no config hash.
    md = run_report_markdown(_summary({"stale": True, "config_hash": None}))
    assert "by the speed command" in md[md.index("## Speed"):]


def test_all_measures_speed_and_resumes(built, tmp_path, small_protocol, capsys):
    layout = RunLayout(_copy(built, tmp_path))
    cfg = RunConfig(suites=("S1",), tools=("oracle", "quick"), scale=0.02, workers=2, threads=2, speed=True)
    assert run_all(cfg, layout) == 0
    record = json.loads(layout.speed_json.read_text())
    manifest = json.loads(layout.manifest.read_text())
    assert record["kind"] == "alignair_assay.speed" and record["config_hash"] == manifest["config_hash"]
    assert record["version"] == speed.SPEED_VERSION == "2"
    assert (record["n_tiny"], record["n_full"], record["repeats"]) == (10, 60, 2)
    assert record["reads"] == {"suite": "S1", "sha256": sha256_file(layout.reads("S1"))}
    # The timed read sets themselves, and every configuration planned, are recorded.
    assert record["sets"] == {"tiny": sha256_file(layout.speed_reads(10)), "full": sha256_file(layout.speed_reads(60))}
    assert record["planned"] == ["quick_1t", "quick_2t"]
    assert set(record["host"]) == HOST_KEYS
    assert list(record["results"]) == ["quick_1t", "quick_2t"]
    result = record["results"]["quick_2t"]
    assert {key: result[key] for key in ("tool", "threads", "cpus", "device")} == {
        "tool": "quick", "threads": 2, "cpus": "0-1", "device": "cpu"}
    assert result["settings"] == {"version": "1"} and result["peak_rss_kb"] == 2048
    assert len(result["walls"]["tiny"]) == len(result["walls"]["full"]) == 2
    assert result["command"] == ["quick", str(layout.speed_reads(60))]
    s1 = read_fasta(layout.reads("S1"))
    assert read_fasta(layout.speed_reads(60)) == [s1[index * len(s1) // 60] for index in range(60)]
    assert (layout.speed_work_dir("quick_2t") / "full" / "predictions.tsv").is_file()

    report = json.loads(layout.report_json.read_text())
    assert report["speed"]["results"] == record["results"] and report["speed"]["failed"] == []
    section = layout.report_md.read_text().split("## Speed")[1]
    assert "| quick_1t |" in section and "| quick_2t |" in section

    # Resume: the measurements are current, so they are kept.
    capsys.readouterr()
    assert run_all(cfg, layout) == 0
    printed = capsys.readouterr().out
    assert "speed quick_1t: up to date, skipped" in printed and "speed quick_2t: up to date, skipped" in printed
    assert json.loads(layout.speed_json.read_text())["results"] == record["results"]

    # A speed.json of another configuration is named in the report, not shown.
    layout.speed_json.write_text(json.dumps({**record, "config_hash": "d" * 64}))
    assert main(["report", "--run-dir", str(layout.root)]) == 0
    speed_report = json.loads(layout.report_json.read_text())["speed"]
    assert speed_report["stale"] and speed_report["config_hash"] == "d" * 64
    assert "another configuration" in speed_report["reason"]
    assert "another configuration" in layout.report_md.read_text().split("## Speed")[1]
    # So is one measured on other S1 reads.
    layout.speed_json.write_text(json.dumps({**record, "reads": {"suite": "S1", "sha256": "0" * 64}}))
    assert main(["report", "--run-dir", str(layout.root)]) == 0
    speed_report = json.loads(layout.report_json.read_text())["speed"]
    assert speed_report["stale"] and "reads" in speed_report["reason"]
    section = layout.report_md.read_text().split("## Speed")[1]
    assert "other S1 reads" in section and "| quick_1t |" not in section


def test_speed_reused_only_on_the_same_protocol_and_read_sets(built, tmp_path, small_protocol, capsys):
    layout = RunLayout(_copy(built, tmp_path))
    cfg = RunConfig(suites=("S1",), tools=("oracle", "quick"), scale=0.02, workers=2, threads=2, speed=True)
    assert run_all(cfg, layout) == 0
    record = json.loads(layout.speed_json.read_text())
    for changed in ({"version": "1"}, {"sets": {**record["sets"], "tiny": "0" * 64}},
                    {"sets": {**record["sets"], "full": "0" * 64}}):
        # Another protocol version, or measured on other read sets (another sampling of the same S1): not kept.
        layout.speed_json.write_text(json.dumps({**record, **changed}))
        capsys.readouterr()
        assert run_all(cfg, layout) == 0
        printed = capsys.readouterr().out
        assert "speed quick_1t: cold start" in printed and "up to date" not in printed.split("speed quick_1t")[1]
        record = json.loads(layout.speed_json.read_text())
        assert record["version"] == "2" and record["sets"]["tiny"] == sha256_file(layout.speed_reads(10))

    # The report does not show a file of another protocol version either.
    layout.speed_json.write_text(json.dumps({**record, "version": "1"}))
    assert main(["report", "--run-dir", str(layout.root)]) == 0
    speed_report = json.loads(layout.report_json.read_text())["speed"]
    assert speed_report["stale"] and "protocol version 1" in speed_report["reason"]


def test_interrupted_speed_lists_missing_configurations(built, tmp_path, small_protocol, capsys):
    layout = RunLayout(_copy(built, tmp_path))
    cfg = RunConfig(suites=("S1",), tools=("oracle", "quick"), scale=0.02, workers=2, threads=2, speed=True)
    assert run_all(cfg, layout) == 0
    record = json.loads(layout.speed_json.read_text())
    # all --speed stopped after its first configuration: speed.json holds it and the plan.
    layout.speed_json.write_text(json.dumps({**record, "results": {"quick_1t": record["results"]["quick_1t"]}}))
    capsys.readouterr()
    assert main(["report", "--run-dir", str(layout.root)]) == 0
    speed_report = json.loads(layout.report_json.read_text())["speed"]
    assert list(speed_report["results"]) == ["quick_1t"] and speed_report["missing"] == ["quick_2t"]
    section = layout.report_md.read_text().split("## Speed")[1]
    assert "| quick_1t |" in section and "- quick_2t: not measured" in section
    assert "speed quick_2t was planned but not measured" in capsys.readouterr().err
    # Resumed, only the missing configuration is measured.
    assert run_all(cfg, layout) == 0
    printed = capsys.readouterr().out
    assert "speed quick_1t: up to date, skipped" in printed and "speed quick_2t: cold start" in printed
    assert json.loads(layout.report_json.read_text())["speed"]["missing"] == []


class Local(Quick):
    """``quick`` registered as alignair-local: timed on its checkpoint without running AlignAIR."""

    name = "alignair-local"


def test_speed_json_written_before_first_measurement(built, tmp_path, small_protocol, monkeypatch):
    layout = RunLayout(_copy(built, tmp_path))
    cfg = RunConfig(suites=("S1",), tools=("oracle", "quick"), scale=0.02, workers=2, threads=2, speed=True)
    seen = []

    def interrupted(*args, **kwargs):
        seen.append(json.loads(layout.speed_json.read_text()))
        raise KeyboardInterrupt

    with monkeypatch.context() as patched:
        patched.setattr(speed, "measure", interrupted)
        with pytest.raises(KeyboardInterrupt):
            speed.run_speed(cfg, layout, Reference.from_dataconfig(cfg.reference))
    # Already on disk when the first measurement began: the plan and no result.
    assert seen[0]["planned"] == ["quick_1t", "quick_2t"] and seen[0]["results"] == {}
    # What the file holds after the interruption: every configuration planned, none measured (so all are missing).
    record = json.loads(layout.speed_json.read_text())
    assert [name for name in record["planned"] if name not in record["results"]] == ["quick_1t", "quick_2t"]


def test_rows_of_another_checkpoint_are_stale(built, tmp_path, small_protocol, monkeypatch, capsys):
    monkeypatch.setitem(RUNNERS, "alignair-local", Local)
    monkeypatch.setattr(speed, "local_cuda_available", lambda: True)
    checkpoint, other = tmp_path / "a.alignair", tmp_path / "b.alignair"
    checkpoint.write_bytes(b"weights 1")
    other.write_bytes(b"weights 2")
    layout = RunLayout(_copy(built, tmp_path))
    # alignair-local is timed for its checkpoint but is not one of the compared tools, so the checkpoint is not part
    # of the configuration hash.
    cfg = RunConfig(suites=("S1",), tools=("oracle", "quick"), scale=0.02, workers=2, threads=2, speed=True,
                    alignair_checkpoint=str(checkpoint))
    assert run_all(cfg, layout) == 0
    results = json.loads(layout.speed_json.read_text())["results"]
    local = ["alignair-local_cuda", "alignair-local_2t"]
    assert list(results) == ["quick_1t", "quick_2t", *local]
    assert {results[name]["checkpoint_sha256"] for name in local} == {sha256_file(checkpoint)}
    assert list(json.loads(layout.report_json.read_text())["speed"]["results"]) == list(results)

    def shown_and_stale():
        speed_report = json.loads(layout.report_json.read_text())["speed"]
        return list(speed_report["results"]), [row["name"] for row in speed_report["stale_results"]]

    # The checkpoint retrained in place: its rows are named as stale, not shown; the others still are.
    checkpoint.write_bytes(b"weights 1, retrained")
    capsys.readouterr()
    assert main(["report", "--run-dir", str(layout.root)]) == 0
    assert shown_and_stale() == (["quick_1t", "quick_2t"], local)
    section = layout.report_md.read_text().split("## Speed")[1]
    assert "| alignair-local_cuda |" not in section and "| quick_1t |" in section
    assert "- alignair-local_cuda: not shown, measured on another checkpoint" in section
    assert "alignair-local_cuda was measured on another checkpoint" in capsys.readouterr().err

    # Another checkpoint, without --speed: the same configuration hash, and the same stale rows.
    assert run_all(dataclasses.replace(cfg, alignair_checkpoint=str(other), speed=False), layout) == 0
    assert shown_and_stale() == (["quick_1t", "quick_2t"], local)

    # all --speed with it measures those rows again and keeps the others.
    capsys.readouterr()
    assert run_all(dataclasses.replace(cfg, alignair_checkpoint=str(other)), layout) == 0
    printed = capsys.readouterr().out
    assert "speed quick_1t: up to date, skipped" in printed and "speed alignair-local_2t: cold start" in printed
    assert shown_and_stale() == (["quick_1t", "quick_2t", *local], [])


def test_rows_of_other_local_code_are_stale(built, tmp_path, small_protocol, monkeypatch, capsys):
    # A copy of this checkout's src/alignair stands in for the code the alignair-local CLI runs.
    source = tmp_path / "checkout" / "src"
    shutil.copytree(alignair_module.SOURCE_DIR / "alignair", source / "alignair",
                    ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(alignair_module, "SOURCE_DIR", source)
    monkeypatch.setitem(RUNNERS, "alignair-local", Local)
    monkeypatch.setattr(speed, "local_cuda_available", lambda: False)
    checkpoint = tmp_path / "a.alignair"
    checkpoint.write_bytes(b"weights 1")
    layout = RunLayout(_copy(built, tmp_path))
    cfg = RunConfig(suites=("S1",), tools=("oracle", "quick"), scale=0.02, workers=2, threads=2, speed=True,
                    alignair_checkpoint=str(checkpoint))
    assert run_all(cfg, layout) == 0
    row = json.loads(layout.speed_json.read_text())["results"]["alignair-local_2t"]
    assert row["code_sha256"] == tree_digest(source / "alignair", suffixes=(".py", ".pyx"))

    # The decoder edited, the checkpoint unchanged: the local row is stale, the others are still shown.
    predict = source / "alignair" / "cli" / "predict.py"
    predict.write_text(predict.read_text() + "\n# decoder change\n")
    capsys.readouterr()
    assert main(["report", "--run-dir", str(layout.root)]) == 0
    speed_report = json.loads(layout.report_json.read_text())["speed"]
    assert list(speed_report["results"]) == ["quick_1t", "quick_2t"]
    assert speed_report["stale_results"] == [
        {"name": "alignair-local_2t", "reason": "measured with other AlignAIR code than this checkout's (code_sha256)"}]
    assert "alignair-local_2t was measured with other AlignAIR code" in capsys.readouterr().err
    # all --speed measures it again and keeps the others.
    assert run_all(cfg, layout) == 0
    printed = capsys.readouterr().out
    assert "speed quick_1t: up to date, skipped" in printed and "speed alignair-local_2t: cold start" in printed
    row = json.loads(layout.speed_json.read_text())["results"]["alignair-local_2t"]
    assert row["code_sha256"] == tree_digest(source / "alignair", suffixes=(".py", ".pyx"))


def test_failed_speed_measurement(built, tmp_path, small_protocol, capsys):
    layout = RunLayout(_copy(built, tmp_path))
    cfg = RunConfig(suites=("S1",), tools=("oracle", "brittle"), scale=0.02, workers=2, threads=2, speed=True)
    assert run_all(cfg, layout) == 2
    results = json.loads(layout.speed_json.read_text())["results"]
    assert "return code 137: Killed" in results["brittle_1t"]["error"] and "cold_start_s" not in results["brittle_1t"]
    assert results["brittle_2t"]["cold_start_s"] > 0
    assert "error: speed brittle_1t failed" in capsys.readouterr().err
    assert json.loads(layout.report_json.read_text())["speed"]["failed"] == [
        {"name": "brittle_1t", "error": results["brittle_1t"]["error"]}]
    assert "- brittle_1t failed:" in layout.report_md.read_text()

    # Resume: the failed configuration is measured again, the other one is kept.
    assert run_all(cfg, layout) == 2
    printed = capsys.readouterr().out
    assert "speed brittle_2t: up to date, skipped" in printed and "speed brittle_1t: failed" in printed


def test_speed_command(built, tmp_path, capsys):
    root = _copy(built, tmp_path)
    layout = RunLayout(root)
    args = ["speed", "--run-dir", str(root), "--tools", "quick", "--threads", "2", "--n-tiny", "5", "--n-full", "20",
            "--repeats", "1"]
    assert main(args) == 0
    record = json.loads(layout.speed_json.read_text())
    assert record["config_hash"] is None and (record["n_tiny"], record["n_full"], record["repeats"]) == (5, 20, 1)
    assert list(record["results"]) == ["quick_1t", "quick_2t"]
    printed = capsys.readouterr().out
    assert "## Speed" in printed and "| quick_2t |" in printed

    # Only the named configurations are timed.
    assert main(args + ["--configs", "quick_2t"]) == 0
    assert list(json.loads(layout.speed_json.read_text())["results"]) == ["quick_2t"]
    capsys.readouterr()

    # Input errors exit with status 2 before anything is timed.
    before = layout.speed_json.read_text()
    assert main(args + ["--configs", "quick_9t"]) == 2
    assert "quick_1t, quick_2t" in capsys.readouterr().err
    assert main([*args[:4], "igblast", *args[5:]]) == 2
    assert "--igblast-dir" in capsys.readouterr().err
    assert main([*args[:2], str(tmp_path / "empty"), *args[3:]]) == 2
    assert "S1" in capsys.readouterr().err
    assert layout.speed_json.read_text() == before
    # A run directory made by all is timed by all --speed, against its manifest.
    layout.manifest.write_text("{}")
    assert main(args) == 2
    assert "all --speed" in capsys.readouterr().err
