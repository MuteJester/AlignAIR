"""The run-directory path schema: every path inside a run directory is built here, and nowhere else.

```
RUN/
  manifest.json                       config + config_hash + split + host + git + genairr provenance
  reference/germline.fasta            shared reference given to every tool (+ reference.json)
  suites/<S>/cases.jsonl.gz           one case per line
  suites/<S>/truth.jsonl.gz           header + observable truth per case
  suites/<S>/reads.fasta              presented reads, id = case_id
  suites/<S>/donors.json              suites with donors only
  suites/<S>/build.json               counts, warnings, timing, config hash (written last)
  runs/<tool>/<S>/<arm>/predictions.tsv + run.json + logs/
  runs/<tool>/_work/                  the tool's one-time setup (e.g. IgBLAST's germline databases)
  scores/<S>/<arm>/
  speed/speed.json
  report.json, report.md
```

``RunLayout`` only joins paths: it never reads, writes or creates anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    """Paths of one run directory rooted at ``root``. Suite, tool and arm names must each be a single path
    component (ValueError otherwise), so no name can point outside its directory."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @property
    def manifest(self) -> Path:
        """The run's configuration and provenance."""
        return self.root / "manifest.json"

    @property
    def reference_dir(self) -> Path:
        """Directory of the shared germline reference given to every tool."""
        return self.root / "reference"

    @property
    def reference_fasta(self) -> Path:
        """The shared germline reference as FASTA."""
        return self.reference_dir / "germline.fasta"

    @property
    def reference_json(self) -> Path:
        """Digest, allele counts and source of the shared reference."""
        return self.reference_dir / "reference.json"

    def suite_dir(self, suite: str) -> Path:
        """Directory of suite ``suite``'s cases, truth and reads."""
        return self.root / "suites" / _component(suite)

    def cases(self, suite: str) -> Path:
        """The suite's cases, one per line (gzip-compressed)."""
        return self.suite_dir(suite) / "cases.jsonl.gz"

    def truth(self, suite: str) -> Path:
        """The suite's observable truth with its header line (gzip-compressed)."""
        return self.suite_dir(suite) / "truth.jsonl.gz"

    def reads(self, suite: str) -> Path:
        """The suite's presented reads as FASTA, ids = case ids."""
        return self.suite_dir(suite) / "reads.fasta"

    def donors(self, suite: str) -> Path:
        """The suite's donors (suites with donors only)."""
        return self.suite_dir(suite) / "donors.json"

    def build_json(self, suite: str) -> Path:
        """The suite's build record; written last, so it marks a complete build."""
        return self.suite_dir(suite) / "build.json"

    def run_dir(self, tool: str, suite: str, arm: str) -> Path:
        """Directory of one tool run on one suite and arm."""
        return self.root / "runs" / _component(tool) / _component(suite) / _component(arm)

    def predictions(self, tool: str, suite: str, arm: str) -> Path:
        """The tool's AIRR rearrangement TSV."""
        return self.run_dir(tool, suite, arm) / "predictions.tsv"

    def run_json(self, tool: str, suite: str, arm: str) -> Path:
        """The tool run's record: command, return code, wall time, versions."""
        return self.run_dir(tool, suite, arm) / "run.json"

    def logs_dir(self, tool: str, suite: str, arm: str) -> Path:
        """The tool run's logs and scratch files."""
        return self.run_dir(tool, suite, arm) / "logs"

    def work_dir(self, tool: str) -> Path:
        """The tool's one-time setup shared by its runs (e.g. IgBLAST's germline databases); ``_work`` cannot
        collide with a suite name."""
        return self.root / "runs" / _component(tool) / "_work"

    def scores_dir(self, suite: str, arm: str) -> Path:
        """Scores of every tool on one suite and arm, and their comparison."""
        return self.root / "scores" / _component(suite) / _component(arm)

    @property
    def speed_dir(self) -> Path:
        """Directory of the speed measurements."""
        return self.root / "speed"

    @property
    def speed_json(self) -> Path:
        """The speed measurements."""
        return self.speed_dir / "speed.json"

    @property
    def report_json(self) -> Path:
        """The run report as JSON."""
        return self.root / "report.json"

    @property
    def report_md(self) -> Path:
        """The run report as markdown."""
        return self.root / "report.md"


def _component(name: str) -> str:
    if not name or name in (".", "..") or Path(name).name != name or "\\" in name:
        raise ValueError(f"not a single path component: {name!r}")
    return name
