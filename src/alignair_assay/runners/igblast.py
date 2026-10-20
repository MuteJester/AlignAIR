"""IgBLAST: germline databases built from the shared reference, then ``igblastn`` with AIRR output (``-outfmt 19``).

The runner needs an unpacked IgBLAST release (``bin/igblastn``, ``bin/makeblastdb``,
``optional_file/human_gl.aux`` and ``internal_data/``, found through ``IGDATA``). ``prepare`` builds the full
germline database once per run directory, in ``layout.work_dir("igblast")``; a run restricted to some alleles
builds its own database in its log directory.
"""
from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable
from pathlib import Path

from alignair_assay.core.cases import read_fasta
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import sha256_file
from alignair_assay.core.reference import Reference
from alignair_assay.runners.base import (
    RunnerOptions,
    RunResult,
    error_line,
    failed_run,
    fasta_ids,
    timed_command,
)
from alignair_assay.runners.reference import write_segment_fastas

_SEGMENTS = ("v", "d", "j")
# Files of a database directory: the segment FASTAs and the BLAST database files, then the makeblastdb logs.
_DATABASE_FILES = ("germline_*.fasta", "V.*", "D.*", "J.*")
_DATABASE_LOGS = "makeblastdb_*"


class IgBlastRunner:
    """IgBLAST from the release in ``igblast_dir``, run with ``threads`` threads and, when ``cpus`` is given,
    pinned to those CPUs with ``taskset``."""

    name = "igblast"

    def __init__(self, igblast_dir: Path, threads: int = 8, cpus: str | None = None):
        self.igblast_dir = Path(igblast_dir)
        self.threads = threads
        self.cpus = cpus
        self._reference: Reference | None = None
        self._database: dict | None = None
        self._setup: dict | None = None

    @classmethod
    def from_options(cls, options: RunnerOptions) -> "IgBlastRunner":
        """The registry factory: ``igblast_dir`` (required, ValueError otherwise), ``threads`` and ``cpus``."""
        if not options.igblast_dir:
            raise ValueError("the igblast runner needs --igblast-dir, the directory of an unpacked IgBLAST release")
        return cls(Path(options.igblast_dir), threads=options.threads, cpus=options.cpus)

    @property
    def igblastn(self) -> Path:
        """The ``igblastn`` binary."""
        return self.igblast_dir / "bin" / "igblastn"

    @property
    def makeblastdb(self) -> Path:
        """The ``makeblastdb`` binary."""
        return self.igblast_dir / "bin" / "makeblastdb"

    @property
    def auxiliary_data(self) -> Path:
        """The human auxiliary file (J reading frames and CDR3 ends)."""
        return self.igblast_dir / "optional_file" / "human_gl.aux"

    def version(self) -> str:
        """The text ``igblastn -version`` prints (RuntimeError when it fails)."""
        completed = subprocess.run([str(self.igblastn), "-version"], capture_output=True, text=True,
                                   env={**os.environ, **self._env()})
        if completed.returncode != 0:
            raise RuntimeError(f"{self.igblastn} -version failed with return code {completed.returncode}: "
                               f"{completed.stderr.strip()}")
        return completed.stdout.strip()

    def fingerprint(self) -> dict:
        """The release directory and the SHA-256 of ``igblastn``, ``makeblastdb`` and the auxiliary file, so a
        release upgraded in place (or a ``current`` symlink moved) is another tool. FileNotFoundError for a missing
        release file."""
        self._check_release()
        return {"igblast_dir": str(self.igblast_dir), "igblastn_sha256": sha256_file(self.igblastn),
                "makeblastdb_sha256": sha256_file(self.makeblastdb),
                "auxiliary_data_sha256": sha256_file(self.auxiliary_data)}

    def _check_release(self) -> None:
        for path in (self.igblastn, self.makeblastdb, self.auxiliary_data):
            if not path.is_file():
                raise FileNotFoundError(f"IgBLAST release {self.igblast_dir} has no "
                                        f"{path.relative_to(self.igblast_dir)}")

    def prepare(self, layout: RunLayout, reference: Reference) -> dict:
        """Check the release, then build the full germline database from the shared ``reference`` in
        ``layout.work_dir("igblast")``. Returns the provenance every run then records: the release directory,
        the ``version`` text, the SHA-256 of ``igblastn``, ``makeblastdb`` and the auxiliary file, and the
        ``database`` (see ``run``). A missing release file raises FileNotFoundError."""
        self._check_release()
        setup = {
            "igblast_dir": str(self.igblast_dir),
            "version": self.version(),
            "sha256": {"igblastn": sha256_file(self.igblastn), "makeblastdb": sha256_file(self.makeblastdb),
                       "auxiliary_data": sha256_file(self.auxiliary_data)},
        }
        directory = layout.work_dir(self.name)
        database = _database_record(directory, self.build_database(reference, directory))
        self._reference, self._database, self._setup = reference, database, setup
        return {**setup, "database": database}

    def build_database(self, reference: Reference, directory: Path,
                       alleles: Iterable[str] | None = None) -> list[list[str]]:
        """Write ``germline_{v,d,j}.fasta`` from ``reference`` (restricted to ``alleles`` when given) into
        ``directory`` and build the BLAST databases ``directory/V``, ``D`` and ``J`` with ``makeblastdb
        -parse_seqids -dbtype nucl``. Files of an earlier database there are removed first. Returns the
        commands run; a failed ``makeblastdb`` raises RuntimeError naming its log."""
        directory = Path(directory)
        for pattern in (*_DATABASE_FILES, _DATABASE_LOGS):
            for path in directory.glob(pattern):
                path.unlink()
        commands = []
        for g, fasta in write_segment_fastas(reference, directory, alleles).items():
            argv = [str(self.makeblastdb), "-parse_seqids", "-dbtype", "nucl", "-in", str(fasta),
                    "-out", str(directory / g.upper())]
            log = directory / f"makeblastdb_{g.upper()}.log"
            returncode, _, _ = timed_command(argv, log_path=log)
            if returncode != 0:
                raise RuntimeError(f"makeblastdb failed with return code {returncode} on {fasta}; see {log}")
            commands.append(argv)
        return commands

    def run(self, fasta: Path, out_tsv: Path, *, suite: str, arm: str, log_dir: Path,
            alleles: Iterable[str] | None = None) -> RunResult:
        """Annotate ``fasta`` into the AIRR TSV ``out_tsv`` with::

            IGDATA=<dir> <dir>/bin/igblastn -germline_db_V <db>/V -germline_db_D <db>/D -germline_db_J <db>/J
                -auxiliary_data <dir>/optional_file/human_gl.aux -organism human -ig_seqtype Ig
                -num_threads <threads> -outfmt 19 -query <fasta> -out <tsv>

        ``<db>`` is the database ``prepare`` built, or, when ``alleles`` is given, one built from those alleles
        in ``log_dir/db`` (its ``makeblastdb`` commands come first in ``commands``). The log is
        ``log_dir/igblastn.log``. The provenance is ``prepare``'s: the release, version and binary digests, and
        the ``database`` searched (directory, allele counts per gene letter, SHA-256 of every database file and
        segment FASTA, and the ``makeblastdb`` commands that built it); plus ``threads``, ``cpus`` and, for a
        failed run, the ``error`` line of its log. A restricted database that cannot be built (an allele not in
        the reference, a gene letter left without alleles, a failed ``makeblastdb``) is a failed run
        (``base.failed_run``), not an exception. RuntimeError before ``prepare``.
        """
        if self._setup is None:
            raise RuntimeError("the igblast runner is not prepared: call prepare first")
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / "igblastn.log"
        database, commands = self._database, []
        if alleles is not None:
            restricted = log_dir / "db"
            try:
                database = _database_record(restricted, self.build_database(self._reference, restricted, alleles))
            except (ValueError, RuntimeError) as error:
                return failed_run(self.name, suite=suite, arm=arm, fasta=fasta, out_tsv=out_tsv, log_path=log,
                                  message=f"the restricted germline database could not be built: {error}",
                                  provenance={**self._setup, "threads": self.threads, "cpus": self.cpus})
            commands = list(database["commands"])
        directory = Path(database["directory"])
        argv = [str(self.igblastn),
                "-germline_db_V", str(directory / "V"), "-germline_db_D", str(directory / "D"),
                "-germline_db_J", str(directory / "J"), "-auxiliary_data", str(self.auxiliary_data),
                "-organism", "human", "-ig_seqtype", "Ig", "-num_threads", str(self.threads), "-outfmt", "19",
                "-query", str(fasta), "-out", str(out_tsv)]
        returncode, wall, rss = timed_command(argv, log_path=log, env=self._env(), cpus=self.cpus)
        commands.append(["env", f"IGDATA={self.igblast_dir}", *argv])
        provenance = {**self._setup, "database": database, "threads": self.threads, "cpus": self.cpus}
        if returncode != 0:
            provenance["error"] = error_line(log)
        return RunResult(tool=self.name, suite=suite, arm=arm, predictions=str(out_tsv), n_reads=len(fasta_ids(fasta)),
                         wall_seconds=wall, peak_rss_kb=rss, returncode=returncode, commands=commands,
                         provenance=provenance)

    def _env(self) -> dict[str, str]:
        return {"IGDATA": str(self.igblast_dir)}


def _database_record(directory: Path, commands: list[list[str]]) -> dict:
    """Where a database is, how many alleles per gene letter it holds, the SHA-256 of each of its files and the
    commands that built it."""
    alleles = {g: len(read_fasta(directory / f"germline_{g}.fasta")) for g in _SEGMENTS
               if (directory / f"germline_{g}.fasta").is_file()}
    files = sorted({path for pattern in _DATABASE_FILES for path in directory.glob(pattern)})
    return {"directory": str(directory), "alleles": alleles, "files": {p.name: sha256_file(p) for p in files},
            "commands": commands}
