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
  runs/<tool>/<S>/genotype/donors/    per donor: reads.<subject>.fasta, predictions.<subject>.tsv (+ logs/<subject>/)
  runs/<tool>/_work/                  the tool's one-time setup (e.g. IgBLAST's germline databases)
  scores/<S>/<arm>/                   <tool>.scores.jsonl.gz, <tool>.report.{json,md}, comparison.{json,md}
  speed/speed.json                    the speed measurements
  speed/reads_<n>.fasta               n reads evenly spaced across S1, timed
  speed/runs/<config>/                each timed configuration's last tiny and full run (predictions, logs)
  report.json, report.md
  diagnostics/<model_key>/            one model's diagnostics (``diagnose``, ``workflow.diagnose``)
  diagnostics/<model_key>/<S>/suite.json   one suite's cached diagnostics, with the key they were computed for
  diagnostics/<model_key>/<S>/report.md    one suite's detail of the last diagnostics report
  diagnostics/<model_key>/profile.json     the cached speed profile, with its key
  diagnostics/<model_key>/profile.md       the speed profile of the last diagnostics report
  diagnostics/<model_key>/diagnostics.json, diagnostics.md   the last diagnostics report and its one-page summary
  real/<donor>/genotype.json          a real donor's genotype, with the genes its genomic contigs capture
  real/<donor>/reads.fasta            the donor's preprocessed unique sequences (+ reads.tsv, one row each)
  real/<donor>/preprocess.json        the preprocessing record, written last (+ preprocess_logs/; preprocess_work/
                                      while it runs)
  real/manifest.json                  the real-data run's configuration (``workflow.real``)
  real/real_summary.json              every donor's and the pooled results; real/real_report.md renders them
  real/_setup/                        the tools' one-time setup (a RunLayout root given to Runner.prepare)
  real/<donor>/fetch.json             what the donor's reads and genotype files were fetched as
  real/<donor>/truth.jsonl.gz         the per-read truth, its record as the header line
  real/<donor>/runs/<tool>/<arm>/     predictions.tsv + run.json (written last) + logs/
  real/<donor>/scores/<arm>/          <tool>.scores.jsonl.gz (each read's classes), <a>_vs_<b>.json comparisons
```

``RunLayout`` only joins paths: it never reads, writes or creates anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    """Paths of one run directory rooted at ``root``. Suite, tool, arm and donor names must each be a single path
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

    def donor_reads(self, tool: str, suite: str, arm: str, subject: str) -> Path:
        """One donor's reads of the suite, as given to the tool in a per-donor arm (``genotype``)."""
        return self.run_dir(tool, suite, arm) / "donors" / f"reads.{_component(subject)}.fasta"

    def donor_predictions(self, tool: str, suite: str, arm: str, subject: str) -> Path:
        """The tool's AIRR TSV for one donor's reads, before the donors are merged into ``predictions``."""
        return self.run_dir(tool, suite, arm) / "donors" / f"predictions.{_component(subject)}.tsv"

    def donor_logs_dir(self, tool: str, suite: str, arm: str, subject: str) -> Path:
        """The logs and scratch files of the tool's run on one donor's reads."""
        return self.logs_dir(tool, suite, arm) / _component(subject)

    def work_dir(self, tool: str) -> Path:
        """The tool's one-time setup shared by its runs (e.g. IgBLAST's germline databases); ``_work`` cannot
        collide with a suite name."""
        return self.root / "runs" / _component(tool) / "_work"

    def scores_dir(self, suite: str, arm: str) -> Path:
        """Scores of every tool on one suite and arm, and their comparison."""
        return self.root / "scores" / _component(suite) / _component(arm)

    def tool_scores(self, suite: str, arm: str, tool: str) -> Path:
        """The tool's per-case scores on one suite and arm (gzip-compressed JSON lines)."""
        return self.scores_dir(suite, arm) / f"{_component(tool)}.scores.jsonl.gz"

    def tool_report_json(self, suite: str, arm: str, tool: str) -> Path:
        """The tool's score report on one suite and arm."""
        return self.scores_dir(suite, arm) / f"{_component(tool)}.report.json"

    def tool_report_md(self, suite: str, arm: str, tool: str) -> Path:
        """The tool's score report on one suite and arm, as markdown."""
        return self.scores_dir(suite, arm) / f"{_component(tool)}.report.md"

    def comparison_json(self, suite: str, arm: str) -> Path:
        """The paired comparison of the two tools on one suite and arm."""
        return self.scores_dir(suite, arm) / "comparison.json"

    def comparison_md(self, suite: str, arm: str) -> Path:
        """The paired comparison of the two tools on one suite and arm, as markdown."""
        return self.scores_dir(suite, arm) / "comparison.md"

    @property
    def speed_dir(self) -> Path:
        """Directory of the speed measurements."""
        return self.root / "speed"

    @property
    def speed_json(self) -> Path:
        """The speed measurements."""
        return self.speed_dir / "speed.json"

    def speed_reads(self, n: int) -> Path:
        """The speed protocol's set of ``n`` reads sampled evenly across S1; ValueError unless ``n`` is a positive
        integer."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"a speed read set needs a positive number of reads, got {n!r}")
        return self.speed_dir / f"reads_{n}.fasta"

    def speed_work_dir(self, config: str) -> Path:
        """The last tiny and full run (predictions and logs) of one timed configuration, e.g. ``igblast_8t``."""
        return self.speed_dir / "runs" / _component(config)

    @property
    def report_json(self) -> Path:
        """The run report as JSON."""
        return self.root / "report.json"

    @property
    def report_md(self) -> Path:
        """The run report as markdown."""
        return self.root / "report.md"

    @property
    def diagnostics_root(self) -> Path:
        """Directory of the model diagnostics, one subdirectory per model key."""
        return self.root / "diagnostics"

    def diagnostics_dir(self, model_key: str) -> Path:
        """The diagnostics of one model, named by ``workflow.diagnose``'s model key (its weights and the AlignAIR
        code it runs)."""
        return self.diagnostics_root / _component(model_key)

    def diagnostics_suite_dir(self, model_key: str, suite: str) -> Path:
        """One suite's diagnostics of the model."""
        return self.diagnostics_dir(model_key) / _component(suite)

    def diagnostics_suite_json(self, model_key: str, suite: str) -> Path:
        """One suite's cached diagnostics of the model, with the key they were computed for."""
        return self.diagnostics_suite_dir(model_key, suite) / "suite.json"

    def diagnostics_profile_json(self, model_key: str) -> Path:
        """The model's cached speed profile, with the key it was measured for."""
        return self.diagnostics_dir(model_key) / "profile.json"

    def diagnostics_json(self, model_key: str) -> Path:
        """The model's last diagnostics report as JSON."""
        return self.diagnostics_dir(model_key) / "diagnostics.json"

    def diagnostics_md(self, model_key: str) -> Path:
        """The one-page summary of the model's last diagnostics report, as markdown."""
        return self.diagnostics_dir(model_key) / "diagnostics.md"

    def diagnostics_suite_md(self, model_key: str, suite: str) -> Path:
        """One suite's detail of the model's last diagnostics report, as markdown (linked from the summary)."""
        return self.diagnostics_suite_dir(model_key, suite) / "report.md"

    def diagnostics_profile_md(self, model_key: str) -> Path:
        """The speed profile of the model's last diagnostics report, as markdown (linked from the summary)."""
        return self.diagnostics_dir(model_key) / "profile.md"

    @property
    def real_root(self) -> Path:
        """Directory of the real-data run, one subdirectory per donor."""
        return self.root / "real"

    def real_donor_dir(self, donor: str) -> Path:
        """One real donor's records."""
        return self.real_root / _component(donor)

    def real_genotype_json(self, donor: str) -> Path:
        """The donor's genotype from genomic sequencing, with the genes its contigs capture and how that was
        computed."""
        return self.real_donor_dir(donor) / "genotype.json"

    def real_reads(self, donor: str) -> Path:
        """The donor's preprocessed unique sequences as FASTA, the reads every tool is given."""
        return self.real_donor_dir(donor) / "reads.fasta"

    def real_reads_meta(self, donor: str) -> Path:
        """One row per sequence of ``real_reads``, in its order: isotype, protocol, counts and primer-masked spans."""
        return self.real_donor_dir(donor) / "reads.tsv"

    def real_preprocess_json(self, donor: str) -> Path:
        """The donor's preprocessing record (written last): counts per step, classifier evidence, fingerprint."""
        return self.real_donor_dir(donor) / "preprocess.json"

    def real_preprocess_logs(self, donor: str) -> Path:
        """The pRESTO output of each preprocessing step."""
        return self.real_donor_dir(donor) / "preprocess_logs"

    def real_preprocess_work(self, donor: str) -> Path:
        """The preprocessing container's one writable directory, removed once the outputs are written."""
        return self.real_donor_dir(donor) / "preprocess_work"

    def real_manifest(self) -> Path:
        """The real-data run's configuration and provenance (``workflow.real``)."""
        return self.real_root / "manifest.json"

    def real_summary_json(self) -> Path:
        """Every donor's and the pooled results of the real-data run."""
        return self.real_root / "real_summary.json"

    def real_report_md(self) -> Path:
        """The real-data run's report, rendered from ``real_summary_json``."""
        return self.real_root / "real_report.md"

    def real_setup_dir(self) -> Path:
        """The root the tools' one-time setup is given (``Runner.prepare(RunLayout(real_setup_dir()), ...)``: the
        IgBLAST databases, the AlignAIR model cache and exported references); ``_setup`` is no donor's name."""
        return self.real_root / "_setup"

    def real_fetch_json(self, donor: str) -> Path:
        """What the donor's reads and genotype files were fetched as (the data root's fetch records)."""
        return self.real_donor_dir(donor) / "fetch.json"

    def real_truth(self, donor: str) -> Path:
        """The donor's per-read truth, with the record of what it was computed from as its header line."""
        return self.real_donor_dir(donor) / "truth.jsonl.gz"

    def real_run_dir(self, tool: str, donor: str, arm: str) -> Path:
        """One tool's run on the donor's reads, on one arm (``shared`` or ``genotype``)."""
        return self.real_donor_dir(donor) / "runs" / _component(tool) / _component(arm)

    def real_predictions(self, tool: str, donor: str, arm: str) -> Path:
        """The tool's AIRR rearrangement TSV of the donor's reads."""
        return self.real_run_dir(tool, donor, arm) / "predictions.tsv"

    def real_run_json(self, tool: str, donor: str, arm: str) -> Path:
        """The tool run's record (written last): command, return code, wall time, its inputs' identity."""
        return self.real_run_dir(tool, donor, arm) / "run.json"

    def real_logs_dir(self, tool: str, donor: str, arm: str) -> Path:
        """The tool run's logs and scratch files."""
        return self.real_run_dir(tool, donor, arm) / "logs"

    def real_scores(self, donor: str, arm: str, tool: str) -> Path:
        """One tool's class of every read's calls on one arm, with the record of what they were computed from."""
        return self.real_donor_dir(donor) / "scores" / _component(arm) / f"{_component(tool)}.scores.jsonl.gz"

    def real_comparison_json(self, donor: str, arm: str, pair: str) -> Path:
        """One paired comparison of the donor's reads (``<a>_vs_<b>`` on an arm, or a tool's arm delta)."""
        return self.real_donor_dir(donor) / "scores" / _component(arm) / f"{_component(pair)}.json"


def _component(name: str) -> str:
    if not name or name in (".", "..") or Path(name).name != name or "\\" in name:
        raise ValueError(f"not a single path component: {name!r}")
    return name
