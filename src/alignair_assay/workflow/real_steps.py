"""The real-data workflow's steps (``workflow.real``): what each computes, its record, and when that record is current.

A step is skipped only while its record is current, as in ``all``: the record holds a fingerprint of the step's
inputs (file digests), the code digest of the modules it runs (``code_digest``) and the pinned container, and a step
whose fingerprint differs is redone, with every step after it that reads its output.

- ``fetch`` (``step_fetch``): the donor's genotype files and read pair through ``real.fetch``, whose records under the
  data root are reused while current (pins, recipe, image, file md5s). ``fetch.json`` keeps those records and the
  md5s the later steps are keyed on; ``reused`` when ``real.fetch`` reused everything.
- ``genotype`` (``step_genotype``): the VDJbase CSV loaded and mapped to the shared reference and the IUIS labels, with
  the genes the donor's contigs capture and resolve (``real.locus.donor_capture``, in the pinned samtools container),
  as ``genotype.json``. Its fingerprint: the md5s of the CSV, the BAM, the BED and the OGRDB set, the reference's
  digest and its gapped forms' (``Reference.gapped_digest``: the IUIS map infers labels through them), the IUIS
  map's OGRDB file, ``SAMTOOLS_IMAGE`` and ``GENOTYPE_CODE``.
- ``preprocess`` (``step_preprocess``): ``real.preprocess.preprocess_run``, which reuses its own record while current
  (the mates' md5s, the pinned Immcantation image, its code and parameters, the cap among them). A run the classifier
  refuses, or whose assembly loss is over the line, raises ``PreprocessError`` with its evidence: the workflow stops
  that donor and goes on.
- ``truth`` (``step_truth``): ``real.truth.real_truths`` on the capped preprocessed reads, the very reads every tool is
  given, in their order, and the truth's audit (``real.truth.truth_health``); ``truth.jsonl.gz`` with its record,
  the audit among it (``health``), as the header line. Its fingerprint: the SHA-256s of
  ``reads.fasta``, ``reads.tsv`` and ``genotype.json``, the reference's digest and its gapped forms' (the shared
  anchors place Cys104 through them), the IUIS map's OGRDB file (the reference-exact known answer's genes) and
  ``TRUTH_CODE``.
- ``runs`` (``run_identity``, ``current_run``, ``make_run``): one tool on one arm, ``Runner.run`` on ``reads.fasta``;
  ``run.json`` (``runners.base.write_run_json``) is written last and records the run's ``identity``: the tool's
  ``fingerprint()`` (binary, image and model digests; a checkpoint variant's label too), the reads' SHA-256, the
  shared reference's content digest (a reference changed under its name is another run: a reused run skips
  ``prepare``, whose model-reference check would otherwise refuse it), the digest of the tool's runner modules
  (``runner_code``) and, on the genotype arm, the arm's own inputs: the digest of the genotype's content
  (``genotype_content_digest``: its alleles, captured, resolved, uncalled and dropped genes, without the record's
  provenance) and the SHA-256 of the arm's allele list. So a changed CSV, capture or IUIS map redoes the genotype
  arm and leaves the shared arm alone, while a capture redone by changed code to the same content keeps it.
  Runner code can change how a tool is invoked or how its output is read, so a runner edit redoing its runs is the
  safe default. A run is current when
  ``require_successful_run`` passes, its identity is the current one and its predictions still have their recorded
  SHA-256; a failed run is made again.

The functions that reach Docker, the data root or the network (``check_images``, ``fetch_inputs``,
``capture_genotype``, ``preprocess_run``, ``load_reference``, ``load_iuis``, ``sensitivity_genes``) are module
attributes the workflow calls by name, so a test can stand in for them.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import NamedTuple

from alignair_assay.core.cases import read_fasta
from alignair_assay.core.container import check_image
from alignair_assay.core.jsonl import read_header, read_jsonl, write_jsonl
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import files_digest, sha256_file
from alignair_assay.core.records import digest, read_json, write_json
from alignair_assay.core.reference import Reference
from alignair_assay.real.fetch import (
    PILOT_DONORS,
    cohort_files,
    fetch_genotype_files,
    fetch_manifest,  # noqa: F401 (called by the workflow through this module, so a test can stand in for it)
    fetch_reads,
)
from alignair_assay.real.genotype import (
    IuisMap,
    RealGenotype,
    genotype_from_json,
    genotype_to_json,
    load_vdjbase_csv,
    sd3_genes,
    sd3_missing,
    sd3_novel,
    vdjbase_cohort_sequences,
)
from alignair_assay.real.locus import SAMTOOLS_IMAGE, donor_capture
from alignair_assay.real.preprocess import preprocess_run
from alignair_assay.real.truth import RealTruth, real_truths, truth_from_json, truth_health, truth_to_json
from alignair_assay.runners import parse_tool
from alignair_assay.runners.base import Runner, RunResult, require_successful_run, write_run_json

PACKAGE = Path(__file__).resolve().parents[1]
# The modules each step runs, relative to the package; their code digest is in the step's fingerprint.
FETCH_CODE = ("real/fetch.py",)
GENOTYPE_CODE = ("real/genotype.py", "real/locus.py", "core/container.py")
TRUTH_CODE = ("real/truth.py", "real/genotype.py", "core/reference.py")
RUNNER_CODE = {"igblast": ("runners/base.py", "runners/igblast.py", "runners/reference.py"),
               "alignair": ("runners/base.py", "runners/alignair.py", "runners/reference.py"),
               "alignair-api": ("runners/base.py", "runners/alignair.py", "runners/alignair_api.py",
                                "runners/reference.py")}
# What a genotype-arm run depends on in ``genotype.json``'s genotype: its content, never its provenance (the
# ``source``, with the capture's commands and container names; the record's code digest and fingerprint).
GENOTYPE_CONTENT = ("alleles", "captured", "resolved", "uncalled", "dropped")
# The module a checkpoint variant (``<tool>@<label>``) runs besides its base's.
VARIANT_CODE = "runners/variants.py"
FETCH_KIND = "alignair_assay.real.fetch_inputs"
GENOTYPE_KIND = "alignair_assay.real.genotype"
TRUTH_KIND = "alignair_assay.real.truth"
# The genotype files of a donor, as ``real.fetch.genotype_specs`` names them.
GENOTYPE_FILES = ("vdjbase_csv", "vdjbase_bam", "ogrdb", "gene_coords")


class Fetched(NamedTuple):
    """A donor's fetched inputs: the read pair and its quality alphabet, the genotype files by key, the md5 of each
    (``reads``: the pair's listing md5), the fetch records as JSON, and whether ``real.fetch`` reused all of them."""

    r1: Path
    r2: Path
    quality: str
    files: dict[str, Path]
    md5: dict[str, str]
    record: dict
    reused: bool


# --- the functions that reach Docker, the data root or the network -----------------------------------------------

def check_images(images: Mapping[str, str]) -> dict[str, dict]:
    """Each pinned image (name -> reference), pulled by digest when not local: ``core.container.check_image``'s record
    by name. ContainerError when Docker is not available or an image is not the pinned one."""
    return {name: check_image(image) for name, image in images.items()}


def load_reference(name: str) -> Reference:
    """The shared reference built from GenAIRR dataconfig ``name``, with its IMGT-gapped V alleles."""
    return Reference.from_dataconfig(name)


def load_iuis(ogrdb_json, reference: Reference) -> IuisMap:
    """The IUIS labels of ``reference``'s alleles from the OGRDB germline set ``ogrdb_json``."""
    return IuisMap.from_ogrdb(ogrdb_json, reference)


def sensitivity_genes(data_root) -> dict:
    """The genes the sensitivity analysis leaves out: those touched by the paper's novel alleles (Supplementary Data 3)
    that the whole VDJbase genomic cohort lacks (``real.genotype.vdjbase_cohort_sequences``, ``sd3_missing``,
    ``sd3_genes``), never only a run's donors. ``{"genes", "sd3_alleles", "missing_alleles", "cohort", "files"}``.
    The cohort files are fetched from their pinned URLs when missing (``real.fetch.cohort_files``); FetchError for one
    that cannot be fetched or is not the pinned file."""
    files = cohort_files(data_root)
    sequences, cohort = vdjbase_cohort_sequences(files["vdjbase_api"].path, files["ogrdb_rev9"].path)
    novel = sd3_novel(files["sd3"].path)
    missing = sd3_missing(sequences, novel)
    return {"genes": sorted(sd3_genes(missing)), "sd3_alleles": len(novel), "missing_alleles": len(missing),
            "cohort": cohort, "files": {key: {"path": str(found.path), "md5": found.md5, "url": found.source_url}
                                        for key, found in files.items()}}


def fetch_inputs(donor: str, data_root, *, threads: int | None = None) -> Fetched:
    """Pilot donor ``donor``'s genotype files (``real.fetch.fetch_genotype_files``) and read pair
    (``real.fetch.fetch_reads``), verified. FetchError as those raise it."""
    root = Path(data_root)
    genotypes = fetch_genotype_files(donor, root)
    r1, r2, reads = fetch_reads(PILOT_DONORS[donor], root, threads=threads)
    md5 = {"reads": reads.md5, **{key: record.md5 for key, record in genotypes.items()}}
    record = {"reads": reads.to_json(root), "genotypes": {key: found.to_json(root) for key, found in genotypes.items()}}
    reused = bool(reads.detail.get("reused")) and all(found.detail.get("reused") for found in genotypes.values())
    return Fetched(r1=r1, r2=r2, quality=reads.quality, files={key: found.path for key, found in genotypes.items()},
                   md5=md5, record=record, reused=reused)


def capture_genotype(donor: str, fetched: Fetched, reference: Reference, iuis: IuisMap) -> RealGenotype:
    """The donor's VDJbase genotype mapped to ``reference`` (``real.genotype.load_vdjbase_csv``) with the genes its
    contigs capture and resolve (``real.locus.donor_capture``, in the pinned samtools container)."""
    genotype = load_vdjbase_csv(fetched.files["vdjbase_csv"], donor, reference, iuis)
    return donor_capture(genotype, fetched.files["vdjbase_bam"], fetched.files["gene_coords"], iuis)


# --- the steps ---------------------------------------------------------------------------------------------------

def step_fetch(donor: str, data_root, layout: RunLayout, *, threads: int | None = None) -> tuple[Fetched, dict]:
    """``fetch_inputs``, its record written to ``real_fetch_json``; the step's ``{"status", "seconds"}``."""
    started = time.monotonic()
    fetched = fetch_inputs(donor, data_root, threads=threads)
    write_json(layout.real_fetch_json(donor), {
        "kind": FETCH_KIND, "donor": donor, "run": PILOT_DONORS.get(donor), "quality": fetched.quality,
        "md5": fetched.md5, "fingerprint": digest(fetched.md5), "code_sha256": code_digest(FETCH_CODE),
        "records": fetched.record})
    return fetched, _step("reused" if fetched.reused else "done", started)


def step_genotype(donor: str, fetched: Fetched, layout: RunLayout, reference: Reference,
                  iuis: IuisMap) -> tuple[RealGenotype, dict]:
    """The donor's captured genotype: read from ``genotype.json`` while its fingerprint is current, else
    ``capture_genotype`` and written there."""
    started = time.monotonic()
    inputs = {**{key: fetched.md5[key] for key in GENOTYPE_FILES}, "reference_digest": reference.digest(),
              "reference_gapped_digest": reference.gapped_digest(), "iuis_ogrdb_sha256": iuis.source.get("sha256")}
    fingerprint = digest({"inputs": inputs, "image": SAMTOOLS_IMAGE, "code_sha256": code_digest(GENOTYPE_CODE)})
    path = layout.real_genotype_json(donor)
    record = read_json(path)
    if record and record.get("kind") == GENOTYPE_KIND and record.get("fingerprint") == fingerprint:
        try:
            return genotype_from_json(record["genotype"]), _step("reused", started)
        except (KeyError, ValueError):
            pass  # unreadable: computed again
    genotype = capture_genotype(donor, fetched, reference, iuis)
    write_json(path, {"kind": GENOTYPE_KIND, "donor": donor, "fingerprint": fingerprint, "inputs": inputs,
                      "image": SAMTOOLS_IMAGE, "code_sha256": code_digest(GENOTYPE_CODE),
                      "genotype": genotype_to_json(genotype)})
    return genotype, _step("done", started)


def step_preprocess(donor: str, fetched: Fetched, layout: RunLayout, *, cap: int, paper_length_filter: bool,
                    accept_assembly_loss: bool, nproc: int) -> tuple[dict, dict]:
    """``preprocess_run`` on the fetched pair, capped at ``cap``; its record and the step's. PreprocessError (with
    its ``evidence``) for a run refused or rejected; its record is written either way."""
    started = time.monotonic()
    record = preprocess_run(donor, fetched.r1, fetched.r2, layout, quality=fetched.quality, cap=cap,
                            paper_length_filter=paper_length_filter, accept_assembly_loss=accept_assembly_loss,
                            nproc=nproc)
    return record, _step("reused" if record.get("reused") else "done", started)


def step_truth(donor: str, layout: RunLayout, genotype: RealGenotype, reference: Reference,
               iuis: IuisMap) -> tuple[list[RealTruth], dict, dict]:
    """The truth of every preprocessed read, in ``reads.fasta``'s order, and its audit (``truth_health``, with the
    IUIS map's names): read from ``truth.jsonl.gz`` while its fingerprint is current, else ``real_truths`` and
    ``truth_health`` and written there (atomically, the record and the audit as the header); and the step's."""
    started = time.monotonic()
    reads, meta = layout.real_reads(donor), layout.real_reads_meta(donor)
    inputs = {"reads_sha256": sha256_file(reads), "meta_sha256": sha256_file(meta),
              "genotype_sha256": sha256_file(layout.real_genotype_json(donor)), "reference_digest": reference.digest(),
              "reference_gapped_digest": reference.gapped_digest(), "iuis_ogrdb_sha256": iuis.source.get("sha256")}
    fingerprint = digest({"inputs": inputs, "code_sha256": code_digest(TRUTH_CODE)})
    path = layout.real_truth(donor)
    header = jsonl_header(path)
    if header.get("fingerprint") == fingerprint and isinstance(header.get("health"), dict):
        return [truth_from_json(row) for row in read_jsonl(path)], header["health"], _step("reused", started)
    sequences, rows = dict(read_fasta(reads)), read_meta(meta)
    truths = real_truths(sequences, rows, genotype, reference, iuis=iuis)
    health = truth_health(truths, genotype, iuis)
    temporary = path.with_name(f".{path.name}")
    write_jsonl((truth_to_json(truth) for truth in truths), temporary,
                header={"kind": TRUTH_KIND, "donor": donor, "fingerprint": fingerprint, "inputs": inputs,
                        "code_sha256": code_digest(TRUTH_CODE), "reads": len(truths), "health": health})
    temporary.replace(path)
    return truths, health, _step("done", started)


def run_identity(tool: str, reads: Path, settings: dict, *, reference_digest: str,
                 alleles: Iterable[str] | None = None, genotype_json: Path | None = None) -> dict:
    """What a run's predictions depend on (the module docstring), as plain JSON data: ``reference_digest`` is the
    shared reference's ``Reference.digest()``."""
    arm = None
    if alleles is not None:
        names = sorted(alleles)
        arm = {"genotype_content_sha256": genotype_content_digest(genotype_json),
               "alleles_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(), "alleles": len(names)}
    identity = {"settings": settings, "reads_sha256": sha256_file(reads), "reference_digest": reference_digest,
                "code_sha256": code_digest(runner_code(tool)), "arm": arm}
    return json.loads(json.dumps(identity))


def genotype_content_digest(genotype_json: Path) -> str:
    """``core.records.digest`` of the ``GENOTYPE_CONTENT`` fields of the genotype a ``genotype.json`` record holds
    (``step_genotype``): what the donor carries and the genes its capture spans, resolves and leaves uncalled,
    without the record's provenance. ValueError for a record without them."""
    record = json.loads(Path(genotype_json).read_text(encoding="utf-8"))
    try:
        genotype = record["genotype"]
        return digest({key: genotype[key] for key in GENOTYPE_CONTENT})
    except (KeyError, TypeError) as error:
        raise ValueError(f"{genotype_json} holds no genotype record ({error!r})") from None


def runner_code(tool: str) -> tuple[str, ...]:
    """The modules behind tool ``tool``'s runs, relative to the package: ``RUNNER_CODE`` of its base
    (``runners/base.py`` alone for a tool not listed), and ``VARIANT_CODE`` too for a checkpoint variant
    (``<base>@<label>``)."""
    base, label = parse_tool(tool)
    files = RUNNER_CODE.get(base, ("runners/base.py",))
    return files if label is None else (*files, VARIANT_CODE)


def current_run(layout: RunLayout, tool: str, donor: str, arm: str, identity: dict) -> RunResult | None:
    """The run's record when it may be reused (the module docstring); None otherwise."""
    try:
        result = require_successful_run(layout.real_run_json(tool, donor, arm))
    except (FileNotFoundError, ValueError, TypeError):
        return None
    predictions = layout.real_predictions(tool, donor, arm)
    provenance = result.provenance
    if provenance.get("identity") != identity or not predictions.is_file():
        return None
    return result if sha256_file(predictions) == provenance.get("predictions_sha256") else None


def make_run(runner: Runner, tool: str, donor: str, arm: str, layout: RunLayout, identity: dict, *,
             alleles: Iterable[str] | None = None) -> RunResult:
    """Run the prepared ``runner`` on the donor's reads into the layout's run directory and record it: the old record
    and predictions are removed first and ``run.json`` is written last, with the ``identity``, the reads and the
    predictions' SHA-256 added to the provenance. A run that raises leaves no record; a failed one is recorded."""
    run_json, predictions = layout.real_run_json(tool, donor, arm), layout.real_predictions(tool, donor, arm)
    run_json.unlink(missing_ok=True)
    predictions.unlink(missing_ok=True)
    predictions.parent.mkdir(parents=True, exist_ok=True)
    reads = layout.real_reads(donor)
    result = runner.run(reads, predictions, suite=donor, arm=arm, log_dir=layout.real_logs_dir(tool, donor, arm),
                        alleles=None if alleles is None else list(alleles))
    provenance = {**result.provenance, "identity": identity,
                  "reads": {"path": str(reads), "sha256": identity["reads_sha256"]},
                  "predictions_sha256": sha256_file(predictions) if predictions.is_file() else None}
    result = dataclasses.replace(result, provenance=provenance)
    write_run_json(result, run_json)
    return result


# --- helpers -----------------------------------------------------------------------------------------------------

def code_digest(paths: Iterable[str]) -> str:
    """``core.provenance.files_digest`` of the package modules ``paths`` (relative to the package)."""
    return files_digest(PACKAGE, [PACKAGE / path for path in paths])


def read_meta(path: Path) -> list[dict]:
    """The rows of a preprocessed run's ``reads.tsv``, in order."""
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def jsonl_header(path: Path) -> dict:
    """The header line of a JSON-lines file written here; {} when it is missing or unreadable."""
    try:
        return read_header(path) or {}
    except (OSError, ValueError, EOFError):
        return {}


def _step(status: str, started: float) -> dict:
    return {"status": status, "seconds": round(time.monotonic() - started, 2)}
