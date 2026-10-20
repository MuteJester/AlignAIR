"""AlignAIR: the released Docker image (``alignair``) and a model checkpoint run with this checkout's CLI
(``alignair-local``).

Both annotate a FASTA with ``alignair predict`` into an AIRR TSV. ``predict`` also writes a ``<out>.run.json``
sidecar, which the runner moves to ``alignair.run.json`` in the log directory and records. Given ``alleles`` (the
genotype arm), a runner writes them as a genotype JSON, ``{"v": [...], "d": [...], "j": [...]}``: the gene-letter
keys that AlignAIR's ``genotype.constraint.load_genotype`` reads. It passes the file as ``--genotype`` with the
default ``mask`` method, which zeroes every other allele. ``prepare`` exports the model's germline reference and
refuses a model whose reference is not the truth reference. AlignAIR has no thread option, so ``threads`` sets
``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import git_state, sha256_file
from alignair_assay.core.reference import Reference
from alignair_assay.runners.base import (
    RunnerOptions,
    RunResult,
    absolute_path,
    error_line,
    failed_run,
    fasta_ids,
    timed_command,
)
from alignair_assay.runners.reference import verify_reference

DEFAULT_IMAGE = "alignair/alignair:3.0.0@sha256:5df66a34ac7b2b3ece927c49ef440c89dc30a47d028415d924c827b6012ccfd3"
DEFAULT_MODEL = "alignair-igh-human@1.0.0"
# The image's model cache; the runner mounts its cache directory there.
CONTAINER_CACHE = "/home/appuser/.cache/alignair"
# The container's shell script for a measured run. It runs alignair on the arguments, prints the container's peak
# memory (cgroup v2, bytes) as the last line of output, and exits with alignair's status. alignair's stderr joins
# its stdout, so the peak stays last; cat's own error is discarded, so it can never become the run's error line.
MEASURED_SCRIPT = 'alignair "$@" 2>&1; rc=$?; cat /sys/fs/cgroup/memory.peak 2>/dev/null; exit $rc'
# The checkout's CLI, run by the interpreter running the assay.
CLI_ENTRY = "import sys; from alignair.cli.main import main; sys.exit(main())"
# Prints the torch version and the name of the CUDA device ``sys.argv[1]`` (None for a CPU device) as JSON.
TORCH_PROBE = ("import json, sys, torch; d = torch.device(sys.argv[1]); "
               "name = torch.cuda.get_device_name(d) if d.type == 'cuda' and torch.cuda.is_available() else None; "
               "print(json.dumps({'torch': torch.__version__, 'cuda_device': name}))")
# The src directory holding alignair_assay, and with it this checkout's alignair package.
SOURCE_DIR = Path(__file__).resolve().parents[2]
LOG = "alignair.log"
SIDECAR = "alignair.run.json"
GENOTYPE = "genotype.json"
# The image runner's staging directory in a run's log directory: the only host directory mounted at /in.
STAGE = "input"
PEAK_RSS = "the container's cgroup v2 memory.peak (resident memory plus page cache)"


def write_genotype_json(reference: Reference, alleles: Iterable[str], path) -> dict[str, list[str]]:
    """Write ``alleles`` as an AlignAIR genotype JSON and return it: per gene letter of ``reference`` (``"v"``,
    ``"d"``, ``"j"``), the sorted alleles. ValueError, before anything is written, for an allele that is not in
    ``reference`` or a gene letter left with no allele (AlignAIR would leave that segment unrestricted)."""
    genotype: dict[str, set[str]] = {letter: set() for letter in reference.alleles}
    for name in alleles:
        if not reference.has(name):
            raise ValueError(f"allele {name!r} is not in the reference")
        genotype[reference.gene_letter(name)].add(name)
    empty = [letter for letter, names in genotype.items() if not names]
    if empty:
        raise ValueError(f"the selected alleles include no {empty[0]} allele")
    written = {letter: sorted(names) for letter, names in genotype.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(written, indent=2) + "\n", encoding="utf-8")
    return written


def memory_peak_kb(log_path) -> int | None:
    """The container's peak memory in KB, read from the last non-empty line of a ``MEASURED_SCRIPT`` log, which is
    cgroup v2 ``memory.peak`` in bytes. None when that line is not a number (the file was unreadable, so alignair's
    own output ends the log)."""
    try:
        lines = [line.strip() for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()]
    except OSError:
        return None
    lines = [line for line in lines if line]
    return int(lines[-1]) // 1024 if lines and lines[-1].isdigit() else None


def host_cache_dir() -> Path:
    """The user's AlignAIR model cache, located as AlignAIR does on Linux: ``ALIGNAIR_CACHE_DIR``, else
    ``$XDG_CACHE_HOME/alignair``, else ``~/.cache/alignair``. The runners only read it."""
    if os.environ.get("ALIGNAIR_CACHE_DIR"):
        return Path(os.environ["ALIGNAIR_CACHE_DIR"])
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "alignair"


class AlignAIRImage:
    """The AlignAIR Docker ``image`` running registry model ``model`` (``id@version``). The model cache is
    ``cache_dir``; by default ``prepare`` uses ``runs/alignair/_work/cache`` of the run directory. ``threads``
    sets the math threads, and ``cpus``, when given, pins the container to a CPU list (``--cpuset-cpus``)."""

    name = "alignair"

    def __init__(self, image: str = DEFAULT_IMAGE, model: str = DEFAULT_MODEL, cache_dir=None,
                 threads: int | None = None, cpus: str | None = None):
        self.image = image
        self.model = model
        self.cache_dir = None if cache_dir is None else Path(absolute_path(cache_dir))
        self.threads = threads
        self.cpus = cpus
        self._model_path: str | None = None
        self._installed: dict | None = None
        self._setup_commands: list[list[str]] = []
        self._reference: Reference | None = None
        self._setup: dict | None = None

    @classmethod
    def from_options(cls, options: RunnerOptions) -> "AlignAIRImage":
        """The registry factory: ``alignair_image`` (default ``DEFAULT_IMAGE``), ``alignair_model``,
        ``cache_dir``, ``threads`` and ``cpus``."""
        return cls(options.alignair_image or DEFAULT_IMAGE, model=options.alignair_model,
                   cache_dir=options.cache_dir, threads=options.threads, cpus=options.cpus)

    def docker_command(self, args: list[str], *, inputs=None, outputs=None, measured: bool = False) -> list[str]:
        """``docker run --rm`` of the image on alignair arguments ``args``, as the current user, with the cache
        directory mounted at ``CONTAINER_CACHE``. Directory ``inputs`` is mounted read-only at ``/in`` and
        ``outputs`` at ``/out`` when given, followed by the CPU pinning and the thread variables. ``measured`` runs
        ``MEASURED_SCRIPT`` through ``sh`` (peak memory last) instead of the image's entrypoint. RuntimeError
        without a cache directory (give ``cache_dir`` or call ``prepare``)."""
        if self.cache_dir is None:
            raise RuntimeError("the alignair runner has no model cache directory: give cache_dir or call prepare")
        argv = ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
                "-v", f"{self.cache_dir}:{CONTAINER_CACHE}"]
        if inputs is not None:
            argv += ["-v", f"{absolute_path(inputs)}:/in:ro"]
        if outputs is not None:
            argv += ["-v", f"{absolute_path(outputs)}:/out"]
        if self.cpus is not None:
            argv += ["--cpuset-cpus", str(self.cpus)]
        if self.threads is not None:
            argv += ["-e", f"OMP_NUM_THREADS={self.threads}", "-e", f"MKL_NUM_THREADS={self.threads}"]
        if measured:
            return [*argv, "--entrypoint", "sh", self.image, "-c", MEASURED_SCRIPT, "sh", *args]
        return [*argv, self.image, *args]

    def predict_command(self, fasta, out_tsv, genotype=None) -> list[str]:
        """The measured ``docker run`` of ``predict --model <model> --input /in/<fasta> --out /out/<tsv>
        --device cpu --quiet [--genotype /in/<genotype>]``, with the FASTA's directory as ``/in`` and the TSV's as
        ``/out``. ValueError for a genotype file outside the FASTA's directory."""
        fasta, out_tsv = Path(absolute_path(fasta)), Path(absolute_path(out_tsv))
        args = ["predict", "--model", self.model, "--input", f"/in/{fasta.name}", "--out", f"/out/{out_tsv.name}",
                "--device", "cpu", "--quiet"]
        if genotype is not None:
            genotype = Path(absolute_path(genotype))
            if genotype.parent != fasta.parent:
                raise ValueError(f"the genotype file {genotype} must be in the FASTA's directory {fasta.parent}, "
                                 "which is mounted at /in")
            args += ["--genotype", f"/in/{genotype.name}"]
        return self.docker_command(args, inputs=fasta.parent, outputs=out_tsv.parent, measured=True)

    def fingerprint(self) -> dict:
        """The ``image`` reference and the ``image_id`` it resolves to (pulled when not local), the ``model`` spec
        and the SHA-256 of the model file it resolves to (``model_sha256``, after ``install_model``), so a moved
        tag or an unpinned model is another tool. Needs a cache directory (RuntimeError otherwise, or when a docker
        command fails)."""
        if self.cache_dir is None:
            raise RuntimeError("the alignair runner has no model cache directory: give cache_dir or call prepare")
        image = self._inspect_image()
        installed = self.install_model()
        return {"image": self.image, "image_id": image["id"], "model": self.model,
                "model_sha256": installed["model"]["sha256"]}

    def prepare(self, layout: RunLayout, reference: Reference) -> dict:
        """Install the model and record what will run; returns the provenance every run then records. The cache
        directory defaults to ``layout.work_dir("alignair") / "cache"``. The provenance holds the
        ``image``: its reference, ID and RepoDigests (``docker image inspect``, after a ``docker pull`` when the
        image is not local). It also holds the ``cache`` and ``model`` records of ``install_model``, the
        ``model_card`` (``models info``), ``doctor`` (``doctor --json``), the ``model_reference``
        (``export_reference`` into the work directory: path, SHA-256, digest) and the ``setup_commands``. The
        exported reference must equal ``reference``: ValueError naming the first difference otherwise.
        RuntimeError when a docker command fails.
        """
        work = layout.work_dir(self.name)
        if self.cache_dir is None:
            self.cache_dir = work / "cache"
        self._setup_commands, self._model_path = [], None
        image = self._inspect_image()
        installed = self.install_model()
        card = self._capture(["models", "info", self.model])
        doctor = self._capture(["doctor", "--json"])
        exported = self.export_reference(work / "model_reference.fasta")
        verify_reference(exported, reference)
        self._reference = reference
        self._setup = {
            "image": image, **installed, "model_card": card, "doctor": json.loads(doctor),
            "model_reference": {"path": str(exported), "sha256": sha256_file(exported), "digest": reference.digest()},
            "setup_commands": list(self._setup_commands),
        }
        return self._setup

    def install_model(self) -> dict:
        """Put the model in the cache directory and find it; returns ``cache`` and ``model`` records. When the
        model is pinned (``id@version``), the cache directory lacks it and the user's cache (``host_cache_dir``)
        holds it, the model file and its ``.sha256`` are copied from there, so no download is
        needed and the user's cache is never written. Then ``models get`` (which downloads only a missing model),
        ``models verify`` (re-hash against the registry; a mismatch fails) and ``models path``. ``cache`` holds
        the ``dir``, the ``model_source`` (``"cache_dir"``: already there, ``"host_cache"``: copied from
        ``seeded_from``, ``"registry"``: downloaded) and the ``verify`` output. ``model`` holds the ``spec``, the
        ``path`` inside the container and the file's ``sha256``. RuntimeError when a command fails.
        """
        if self.cache_dir is None:
            raise RuntimeError("the alignair runner has no model cache directory: give cache_dir or call prepare")
        model_id, _, version = self.model.partition("@")
        relative = Path("models", model_id, f"{version}.alignair") if version else None
        source, seeded_from = "registry", None
        if relative is not None and (self.cache_dir / relative).is_file():
            source = "cache_dir"
        elif relative is not None and (host_cache_dir() / relative).is_file():
            seeded_from = host_cache_dir() / relative
            (self.cache_dir / relative).parent.mkdir(parents=True, exist_ok=True)
            for suffix in ("", ".sha256"):
                origin = seeded_from.with_name(seeded_from.name + suffix)
                if origin.is_file():
                    shutil.copyfile(origin, (self.cache_dir / relative).with_name(origin.name))
            source = "host_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._capture(["models", "get", self.model])
        verify = self._capture(["models", "verify", model_id])
        self._model_path = self._capture(["models", "path", self.model]).splitlines()[-1].strip()
        host_path = self._host_path(self._model_path)
        self._installed = {
            "cache": {"dir": str(self.cache_dir), "model_source": source,
                      "seeded_from": None if seeded_from is None else str(seeded_from), "verify": verify},
            "model": {"spec": self.model, "path": self._model_path,
                      "sha256": None if host_path is None else sha256_file(host_path)},
        }
        return self._installed

    def export_reference(self, out_fasta) -> Path:
        """Write the model's germline reference to ``out_fasta`` with ``reference export <model file> --fasta``
        and return its absolute path. The model's file path is used because 3.0.0 crashes on a registry id
        there. The model is installed first when needed (``install_model``)."""
        out = Path(absolute_path(out_fasta))
        out.parent.mkdir(parents=True, exist_ok=True)
        if self._model_path is None:
            self.install_model()
        self._capture(["reference", "export", self._model_path, "--fasta", f"/out/{out.name}"], outputs=out.parent)
        return out

    def run(self, fasta: Path, out_tsv: Path, *, suite: str, arm: str, log_dir: Path,
            alleles: Iterable[str] | None = None) -> RunResult:
        """Annotate ``fasta`` into ``out_tsv`` with ``predict_command``; the log is ``log_dir/alignair.log``. A copy
        of the FASTA goes to the emptied ``log_dir/input``, which alone becomes ``/in``, so the container never
        sees the directory the reads came from (a suite's cases and truth). With ``alleles``, the genotype JSON
        (``write_genotype_json``) goes there too; alleles the JSON cannot hold make a failed run
        (``base.failed_run``). The wall time includes the container's start. The peak RSS is the container's
        ``memory.peak``; when it is unreadable it is None and ``rss_note`` says why. The provenance is
        ``prepare``'s plus ``threads``, ``cpus``, ``peak_rss`` (what the RSS measures), ``genotype`` (its path,
        SHA-256 and allele counts per gene letter; None without alleles), the ``sidecar`` (None when
        ``predict`` wrote none) and, for a failed run, the ``error`` line of the log. RuntimeError before
        ``prepare``.
        """
        if self._setup is None:
            raise RuntimeError("the alignair runner is not prepared: call prepare first")
        fasta, out_tsv, log_dir = Path(fasta), Path(out_tsv), Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        out_tsv.parent.mkdir(parents=True, exist_ok=True)  # else docker would create the mount point as root
        log = log_dir / LOG
        provenance = {**self._setup, "threads": self.threads, "cpus": self.cpus, "peak_rss": PEAK_RSS,
                      "genotype": None}
        staged = log_dir / STAGE
        shutil.rmtree(staged, ignore_errors=True)
        staged.mkdir()
        given, genotype = staged / fasta.name, None
        shutil.copyfile(fasta, given)
        if alleles is not None:
            try:
                written = write_genotype_json(self._reference, alleles, staged / GENOTYPE)
            except ValueError as error:
                return failed_run(self.name, suite=suite, arm=arm, fasta=fasta, out_tsv=out_tsv, log_path=log,
                                  message=f"the genotype could not be written: {error}", provenance=provenance)
            genotype = staged / GENOTYPE
            provenance["genotype"] = _genotype_record(genotype, written)
        argv = self.predict_command(given, out_tsv, genotype)
        _sidecar(out_tsv).unlink(missing_ok=True)
        returncode, wall, _ = timed_command(argv, log_path=log)
        # GNU time measured the docker client, not AlignAIR: drop its readout so it is not mistaken for the peak.
        log.with_name(log.name + ".peak_rss_kb").unlink(missing_ok=True)
        peak = memory_peak_kb(log)
        if peak is None:
            provenance["rss_note"] = (f"/sys/fs/cgroup/memory.peak was not readable in the container; the log's "
                                      f"last message is {error_line(log)!r}")
        provenance["sidecar"] = _move_sidecar(out_tsv, log_dir / SIDECAR)
        if returncode != 0:
            provenance["error"] = error_line(log)
        return RunResult(tool=self.name, suite=suite, arm=arm, predictions=str(out_tsv), n_reads=len(fasta_ids(fasta)),
                         wall_seconds=wall, peak_rss_kb=peak, returncode=returncode, commands=[argv],
                         provenance=provenance)

    def _inspect_image(self) -> dict:
        """The image's reference, ID and RepoDigests, pulling it first when it is not local."""
        inspect = ["docker", "image", "inspect", self.image]
        completed = subprocess.run(inspect, capture_output=True, text=True)
        if completed.returncode != 0:
            self._checked(["docker", "pull", self.image])
            completed = subprocess.run(inspect, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"docker cannot find image {self.image}: {completed.stderr.strip()}")
        record = json.loads(completed.stdout)[0]
        return {"reference": self.image, "id": record["Id"], "repo_digests": record.get("RepoDigests") or []}

    def _capture(self, args: list[str], *, outputs=None) -> str:
        """The stripped output of the image on alignair arguments ``args``; the command is kept for provenance."""
        return self._checked(self.docker_command(args, outputs=outputs))

    def _checked(self, argv: list[str]) -> str:
        self._setup_commands.append(argv)
        completed = subprocess.run(argv, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"{' '.join(argv)} failed with return code {completed.returncode}: "
                               f"{(completed.stdout + completed.stderr).strip()[-2000:]}")
        return completed.stdout.strip()

    def _host_path(self, container_path: str) -> Path | None:
        """The host file of a path inside the mounted cache; None for a path outside it."""
        path = Path(container_path)
        if not path.is_relative_to(CONTAINER_CACHE):
            return None
        return self.cache_dir / path.relative_to(CONTAINER_CACHE)


class AlignAIRLocal:
    """A model checkpoint (a ``.alignair`` file) run with this checkout's AlignAIR CLI. It runs in a subprocess of
    the interpreter running the assay, with ``PYTHONPATH`` set to the checkout's ``src``, on torch device
    ``device`` with ``batch_size`` reads per batch. ``threads`` sets the math threads, and ``cpus``, when given,
    pins it with ``taskset``."""

    name = "alignair-local"

    def __init__(self, checkpoint, device: str = "cpu", batch_size: int = 64, threads: int | None = None,
                 cpus: str | None = None):
        self.checkpoint = Path(absolute_path(checkpoint))
        self.device = device
        self.batch_size = batch_size
        self.threads = threads
        self.cpus = cpus
        self._setup_commands: list[list[str]] = []
        self._reference: Reference | None = None
        self._setup: dict | None = None

    @classmethod
    def from_options(cls, options: RunnerOptions) -> "AlignAIRLocal":
        """The registry factory: ``alignair_checkpoint`` (required, ValueError otherwise), ``device``,
        ``batch_size``, ``threads`` and ``cpus``."""
        if not options.alignair_checkpoint:
            raise ValueError("the alignair-local runner needs the path of a .alignair model file: --checkpoint (run) "
                             "or --alignair-checkpoint (all)")
        return cls(options.alignair_checkpoint, device=options.device, batch_size=options.batch_size,
                   threads=options.threads, cpus=options.cpus)

    def environment(self) -> dict[str, str]:
        """The variables the CLI runs with: ``PYTHONPATH`` (the checkout's ``src``) and, with ``threads``,
        ``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS``."""
        env = {"PYTHONPATH": str(SOURCE_DIR)}
        if self.threads is not None:
            env.update(OMP_NUM_THREADS=str(self.threads), MKL_NUM_THREADS=str(self.threads))
        return env

    def predict_command(self, fasta, out_tsv, genotype=None) -> list[str]:
        """``<python> -c CLI_ENTRY predict --model <checkpoint> --input <fasta> --out <tsv> --device <device>
        --batch-size <n> --quiet [--genotype <genotype>]``."""
        argv = [sys.executable, "-c", CLI_ENTRY, "predict", "--model", str(self.checkpoint), "--input", str(fasta),
                "--out", str(out_tsv), "--device", self.device, "--batch-size", str(self.batch_size), "--quiet"]
        return argv if genotype is None else [*argv, "--genotype", str(genotype)]

    def fingerprint(self) -> dict:
        """The ``checkpoint`` path, its SHA-256 (so a model retrained in place is another tool) and the torch
        ``device``. FileNotFoundError for a missing checkpoint."""
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"no AlignAIR checkpoint {self.checkpoint}")
        return {"checkpoint": str(self.checkpoint), "checkpoint_sha256": sha256_file(self.checkpoint),
                "device": self.device}

    def prepare(self, layout: RunLayout, reference: Reference) -> dict:
        """Record what will run; returns the provenance every run then records. The provenance holds the
        ``checkpoint`` (path and SHA-256), the ``repository`` (the checkout's path,
        ``git_commit`` and ``git_dirty``), the ``python`` interpreter, the ``torch`` version and ``cuda_device``
        name the subprocess sees (None on a CPU device), ``device``, ``batch_size``, the ``model_reference``
        (``export_reference`` into ``layout.work_dir("alignair-local")``: path, SHA-256, digest) and the
        ``setup_commands``. The exported reference must equal ``reference``: ValueError naming the first
        difference otherwise. FileNotFoundError for a missing checkpoint; RuntimeError when a command fails.
        """
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"no AlignAIR checkpoint {self.checkpoint}")
        self._setup_commands = []
        torch_info = json.loads(self._checked([sys.executable, "-c", TORCH_PROBE, self.device]))
        exported = self.export_reference(layout.work_dir(self.name) / "model_reference.fasta")
        verify_reference(exported, reference)
        repository = SOURCE_DIR.parent
        self._reference = reference
        self._setup = {
            "checkpoint": {"path": str(self.checkpoint), "sha256": sha256_file(self.checkpoint)},
            "repository": {"path": str(repository), **git_state(repository)},
            "python": sys.executable, **torch_info, "device": self.device, "batch_size": self.batch_size,
            "model_reference": {"path": str(exported), "sha256": sha256_file(exported), "digest": reference.digest()},
            "setup_commands": list(self._setup_commands),
        }
        return self._setup

    def export_reference(self, out_fasta) -> Path:
        """Write the checkpoint's germline reference to ``out_fasta`` with ``reference export <checkpoint>
        --fasta`` and return its absolute path. RuntimeError when the command fails."""
        out = Path(absolute_path(out_fasta))
        out.parent.mkdir(parents=True, exist_ok=True)
        self._checked([sys.executable, "-c", CLI_ENTRY, "reference", "export", str(self.checkpoint),
                       "--fasta", str(out)])
        return out

    def run(self, fasta: Path, out_tsv: Path, *, suite: str, arm: str, log_dir: Path,
            alleles: Iterable[str] | None = None) -> RunResult:
        """Annotate ``fasta`` into ``out_tsv`` with ``predict_command`` under ``timed_command``. The run uses
        ``environment()``, is pinned to ``cpus``, and logs to ``log_dir/alignair.log``. With ``alleles``, the
        genotype JSON (``write_genotype_json``) is ``log_dir/genotype.json``. Alleles the
        JSON cannot hold make a failed run (``base.failed_run``). The peak RSS is the CLI process's (GNU time).
        The recorded command carries the environment as an ``env`` prefix. The provenance is ``prepare``'s plus
        ``threads``, ``cpus``, ``genotype`` (its path, SHA-256 and allele counts per gene letter; None without
        alleles), the ``sidecar`` (None when ``predict`` wrote none) and, for a failed run, the ``error`` line of
        the log. RuntimeError before ``prepare``.
        """
        if self._setup is None:
            raise RuntimeError("the alignair-local runner is not prepared: call prepare first")
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
        argv = self.predict_command(fasta, out_tsv, genotype)
        env = self.environment()
        _sidecar(out_tsv).unlink(missing_ok=True)
        returncode, wall, rss = timed_command(argv, log_path=log, env=env, cpus=self.cpus)
        provenance["sidecar"] = _move_sidecar(out_tsv, log_dir / SIDECAR)
        if returncode != 0:
            provenance["error"] = error_line(log)
        return RunResult(tool=self.name, suite=suite, arm=arm, predictions=str(out_tsv), n_reads=len(fasta_ids(fasta)),
                         wall_seconds=wall, peak_rss_kb=rss, returncode=returncode,
                         commands=[["env", *(f"{key}={value}" for key, value in env.items()), *argv]],
                         provenance=provenance)

    def _checked(self, argv: list[str]) -> str:
        """The stripped stdout of ``argv`` run with ``environment()``; the command is kept for provenance."""
        env = self.environment()
        self._setup_commands.append(["env", *(f"{key}={value}" for key, value in env.items()), *argv])
        completed = subprocess.run(argv, capture_output=True, text=True, env={**os.environ, **env})
        if completed.returncode != 0:
            raise RuntimeError(f"{' '.join(argv)} failed with return code {completed.returncode}: "
                               f"{(completed.stdout + completed.stderr).strip()[-2000:]}")
        return completed.stdout.strip()


def _genotype_record(path: Path, genotype: dict[str, list[str]]) -> dict:
    return {"path": str(path), "sha256": sha256_file(path),
            "alleles": {letter: len(names) for letter, names in genotype.items()}}


def _sidecar(out_tsv: Path) -> Path:
    """Where ``predict --out <tsv>`` writes its run sidecar."""
    return out_tsv.with_name(out_tsv.name + ".run.json")


def _move_sidecar(out_tsv: Path, target: Path) -> dict | None:
    """Move the run sidecar of ``out_tsv`` to ``target`` and return its contents; None when there is none."""
    target.unlink(missing_ok=True)
    source = _sidecar(out_tsv)
    if not source.is_file():
        return None
    os.replace(source, target)
    return json.loads(target.read_text(encoding="utf-8"))
