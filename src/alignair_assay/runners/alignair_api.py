"""AlignAIR through its public Python API (``alignair-api``): a model checkpoint run on both arms by one code path.

AlignAIR 3.0.0's ``predict --genotype`` crashes (``cli/predict.py`` passes ``load_genotype``'s ``(genotype,
unknown)`` pair on as the genotype), and this branch never changes ``src/alignair``. This runner therefore calls
``alignair.Aligner.with_genotype(...).predict(...)`` itself, in a subprocess that runs the diagnostics layer's
``api_predict`` module by name (``DRIVER_MODULE``): runners may not import ``alignair``, and the assay process never
imports torch. The module loads the checkpoint with the diagnostics' loader, reads the FASTA and writes the AIRR TSV
as the CLI does (see its docstring), so on the shared arm its rows are alignair-local's, and the genotype arm runs
the same code and model with the genotype added.

Everything else is alignair-local's (``AlignAIRLocal``): the settings, the environment, ``prepare`` (the torch probe
and the exported reference, refused unless it is the truth reference) and the genotype JSON. A failure of the
genotype path (the module's exit status ``GENOTYPE_API_FAILED``) is recorded with the note ``GENOTYPE_NOTE``, as
design §10.3 asks, so it is told apart from any other failure.
"""
from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path

from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import sha256_file
from alignair_assay.core.reference import Reference
from alignair_assay.runners.alignair import (
    GENOTYPE,
    LOG,
    SOURCE_DIR,
    AlignAIRLocal,
    _genotype_record,
    _move_sidecar,
    _sidecar,
    write_genotype_json,
)
from alignair_assay.runners.base import RunnerOptions, RunResult, error_line, failed_run, fasta_ids, timed_command

# The module the subprocess runs, by name, and its file (read for the fingerprint, never imported here).
DRIVER_MODULE = "alignair_assay.diagnostics.api_predict"
DRIVER = SOURCE_DIR / "alignair_assay" / "diagnostics" / "api_predict.py"
# The module's exit status for a failure on the genotype path, and the note a run records for it (the module's
# GENOTYPE_FAILED and GENOTYPE_NOTE).
GENOTYPE_API_FAILED = 3
GENOTYPE_NOTE = "genotype API path failed"
# Where the module's run record (``<out>.run.json``) is moved to in the log directory.
RECORD = "api_predict.run.json"


class AlignAIRApi(AlignAIRLocal):
    """A model checkpoint (a ``.alignair`` file) run through AlignAIR's public Python API, in a subprocess of the
    interpreter running the assay with ``PYTHONPATH`` set to the checkout's ``src``, on torch device ``device`` with
    ``batch_size`` reads per batch. ``threads`` sets the math threads, and ``cpus``, when given, pins it with
    ``taskset``."""

    name = "alignair-api"

    @classmethod
    def from_options(cls, options: RunnerOptions) -> "AlignAIRApi":
        """The registry factory: ``alignair_checkpoint`` (required, ValueError otherwise), ``device``,
        ``batch_size``, ``threads`` and ``cpus``."""
        if not options.alignair_checkpoint:
            raise ValueError("the alignair-api runner needs the path of a .alignair model file: --checkpoint (run) "
                             "or --alignair-checkpoint (all, speed)")
        return cls(options.alignair_checkpoint, device=options.device, batch_size=options.batch_size,
                   threads=options.threads, cpus=options.cpus)

    def predict_command(self, fasta, out_tsv, genotype=None, *, reference) -> list[str]:
        """``<python> -m DRIVER_MODULE --checkpoint <checkpoint> --reference <reference> --input <fasta> --out <tsv>
        --device <device> --batch-size <n> [--genotype <genotype>]``; ``reference`` is the FASTA the model's
        reference must equal (``prepare``'s exported one, verified against the truth reference)."""
        argv = [sys.executable, "-m", DRIVER_MODULE, "--checkpoint", str(self.checkpoint), "--reference",
                str(reference), "--input", str(fasta), "--out", str(out_tsv), "--device", self.device,
                "--batch-size", str(self.batch_size)]
        return argv if genotype is None else [*argv, "--genotype", str(genotype)]

    def fingerprint(self) -> dict:
        """alignair-local's fingerprint (the checkpoint path and SHA-256, the device, the checkout's
        ``src/alignair`` digest and the torch version) and ``driver_sha256``, the SHA-256 of the module that drives
        the API in the CLI's place, so an edit of how it reads, predicts or writes is another tool.
        FileNotFoundError for a missing checkpoint."""
        return {**super().fingerprint(), "driver_sha256": sha256_file(DRIVER)}

    def prepare(self, layout: RunLayout, reference: Reference) -> dict:
        """alignair-local's ``prepare`` (into ``layout.work_dir("alignair-api")``), whose provenance gains the
        ``driver``: its module name, path and SHA-256."""
        setup = super().prepare(layout, reference)
        setup["driver"] = {"module": DRIVER_MODULE, "path": str(DRIVER), "sha256": sha256_file(DRIVER)}
        return setup

    def run(self, fasta: Path, out_tsv: Path, *, suite: str, arm: str, log_dir: Path,
            alleles: Iterable[str] | None = None) -> RunResult:
        """Annotate ``fasta`` into ``out_tsv`` with ``predict_command`` under ``timed_command``, as alignair-local
        runs the CLI: with ``environment()``, pinned to ``cpus``, logged to ``log_dir/alignair.log``, the peak RSS
        the subprocess's, and with ``alleles`` written as ``log_dir/genotype.json`` (alleles the JSON cannot hold
        make a failed run, ``base.failed_run``, before anything runs). The module's run record is moved to
        ``log_dir/api_predict.run.json`` and kept as the provenance's ``sidecar`` (None when it wrote none). The
        provenance is ``prepare``'s plus ``threads``, ``cpus``, ``genotype`` and, for a failed run, the ``error``
        line of the log; a failure on the genotype path (exit status ``GENOTYPE_API_FAILED``) also gets ``note``
        ``GENOTYPE_NOTE``. RuntimeError before ``prepare``."""
        if self._setup is None:
            raise RuntimeError("the alignair-api runner is not prepared: call prepare first")
        fasta, out_tsv, log_dir = Path(fasta), Path(out_tsv), Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / LOG
        provenance = {**self._setup, "threads": self.threads, "cpus": self.cpus, "genotype": None}
        genotype = None
        if alleles is not None:
            genotype = log_dir / GENOTYPE
            try:
                written = write_genotype_json(self._reference, alleles, genotype)
            except ValueError as error:
                return failed_run(self.name, suite=suite, arm=arm, fasta=fasta, out_tsv=out_tsv, log_path=log,
                                  message=f"the genotype could not be written: {error}", provenance=provenance)
            provenance["genotype"] = _genotype_record(genotype, written)
        argv = self.predict_command(fasta, out_tsv, genotype, reference=self._setup["model_reference"]["path"])
        env = self.environment()
        _sidecar(out_tsv).unlink(missing_ok=True)
        returncode, wall, rss = timed_command(argv, log_path=log, env=env, cpus=self.cpus)
        provenance["sidecar"] = _move_sidecar(out_tsv, log_dir / RECORD)
        if returncode != 0:
            provenance["error"] = error_line(log)
        if returncode == GENOTYPE_API_FAILED and genotype is not None:
            provenance["note"] = GENOTYPE_NOTE
        return RunResult(tool=self.name, suite=suite, arm=arm, predictions=str(out_tsv), n_reads=len(fasta_ids(fasta)),
                         wall_seconds=wall, peak_rss_kb=rss, returncode=returncode,
                         commands=[["env", *(f"{key}={value}" for key, value in env.items()), *argv]],
                         provenance=provenance)
