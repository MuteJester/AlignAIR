"""The oracle: a runner that answers every read with the perfect prediction, read from the suite's truth.

It runs in-process, so it has no command and no peak RSS. It is the check that a suite's truth and the scoring
pipeline agree: through the full TSV round trip it must score 1.0 on every compared metric.
"""
from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

from alignair_assay.airr.writer import write_airr_tsv
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import sha256_file
from alignair_assay.core.reference import Reference
from alignair_assay.runners.base import RunnerOptions, RunResult, fasta_ids
from alignair_assay.truth.io import read_truth_jsonl
from alignair_assay.truth.oracle import oracle_prediction


class OracleRunner:
    """Writes ``truth.oracle.oracle_prediction`` rows for the reads of a FASTA, from ``layout.truth(suite)`` of
    the run directory it is bound to: given at construction, or by ``prepare``."""

    name = "oracle"

    def __init__(self, layout: RunLayout | None = None):
        self.layout = layout

    @classmethod
    def from_options(cls, options: RunnerOptions) -> "OracleRunner":
        """The registry factory: the oracle has no options, and ``prepare`` binds the run directory."""
        return cls()

    def fingerprint(self) -> dict:
        """Nothing: the oracle answers from the suite's truth, whose SHA-256 each of its runs records."""
        return {}

    def prepare(self, layout: RunLayout, reference: Reference) -> dict:
        """Bind the run directory whose truth files the oracle reads; nothing to set up, so no provenance."""
        self.layout = layout
        return {}

    def run(self, fasta: Path, out_tsv: Path, *, suite: str, arm: str, log_dir: Path,
            alleles: Iterable[str] | None = None) -> RunResult:
        """Write one oracle row per read of ``fasta``, in its order, to ``out_tsv``. A read id that is not a case
        of the suite's truth raises ValueError; ``alleles`` is ignored, since the truth needs no restriction.
        The provenance holds the truth file's path and SHA-256."""
        if self.layout is None:
            raise RuntimeError("the oracle runner has no run directory: call prepare first")
        start = time.perf_counter()
        truth_path = self.layout.truth(suite)
        truths = {t.case_id: t for t in read_truth_jsonl(truth_path)}
        ids = fasta_ids(fasta)
        unknown = [cid for cid in ids if cid not in truths]
        if unknown:
            raise ValueError(f"{fasta}: read {unknown[0]!r} is not a case of {truth_path} ({len(unknown)} unknown)")
        write_airr_tsv((oracle_prediction(truths[cid]) for cid in ids), out_tsv)
        return RunResult(tool=self.name, suite=suite, arm=arm, predictions=str(out_tsv), n_reads=len(ids),
                         wall_seconds=time.perf_counter() - start, peak_rss_kb=None, returncode=0, commands=[],
                         provenance={"truth": {"path": str(truth_path), "sha256": sha256_file(truth_path)}})
