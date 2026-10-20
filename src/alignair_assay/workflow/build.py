"""Build one suite into a run directory: generate its cases, build their truth, write reads and the build record.

The files are ``core.layout.RunLayout``'s ``cases``, ``truth``, ``reads``, ``donors`` (suites with donors only)
and ``build_json`` of the suite; ``build.json`` is removed first and written last, so it exists only for a
complete build. A build is current while its recorded identity (``build_identity``) is the configuration's: the
suite, split, reference, seed and scale, the source of the code that generates and truths it, the truth version and
the GenAIRR in use.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from alignair_assay.core.cases import write_cases, write_fasta
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import files_digest, genairr_provenance
from alignair_assay.core.reference import Reference
from alignair_assay.suites import catalog_module, get_suite
from alignair_assay.suites.donors import donors_summary, donors_to_json
from alignair_assay.suites.engine import check_split, generate_suite
from alignair_assay.truth.case import truth_from_case
from alignair_assay.truth.io import TRUTH_KIND, TRUTH_VERSION, read_truth_header, write_truth_jsonl
from alignair_assay.truth.summary import truth_summary

# The alignair_assay package directory; source files are named relative to it.
PACKAGE_DIR = Path(__file__).resolve().parents[1]
# What every suite's cases depend on besides its catalog module: the suite types, the generator, donors, shared
# recipes, novel alleles, and the case schema and presentation.
SUITE_SOURCES = ("suites/base.py", "suites/engine.py", "suites/donors.py", "suites/recipes.py", "suites/novel.py",
                 "core/cases.py", "core/seqops.py")
# What the truth depends on besides the truth package itself.
TRUTH_SOURCES = ("core/cigar.py", "core/genotype.py", "core/reference.py", "core/seqops.py")
# The GenAIRR provenance fields that identify the simulator (``core.provenance.genairr_provenance``).
GENAIRR_IDENTITY = ("genairr_version", "genairr_git_commit", "genairr_git_dirty", "genairr_source_sha256")


def build_suite(name: str, layout: RunLayout, *, reference_name: str, split: str, seed: int, scale: float,
                workers: int) -> dict:
    """Generate suite ``name`` (``suites.engine.generate_suite``) and write its files; returns the build record.

    The truth header holds ``kind``, ``version``, ``reference``, ``reference_digest``, the GenAIRR provenance,
    ``suite``, ``split``, ``seed``, ``scale`` and ``n_cases``. The build record (``build.json``) holds the same
    configuration, its ``identity`` (``build_identity``) and ``config_hash``, the per-stratum generation stats
    (``strata``), the truth summary (``truth``: excluded cases, warnings, unscorable genes, simulator calls, primary
    fallbacks), ``workers`` and the wall time in ``seconds``. A suite with donors also gets ``donors.json``
    (``suites.donors.donors_to_json``) and a ``donors_summary`` (``suites.donors.donors_summary``: realised alleles
    per gene, deleted genes per donor, heterozygous share) in its build record, and each of its cases is truthed
    with the novel alleles of its donor (``groups["subject"]``). Cases, truth and donors do not depend on
    ``workers``.
    """
    start = time.perf_counter()
    suite = get_suite(name)
    check_split(split)
    seed, scale = int(seed), float(scale)
    layout.build_json(name).unlink(missing_ok=True)
    layout.donors(name).unlink(missing_ok=True)
    reference = Reference.from_dataconfig(reference_name)
    digest = reference.digest()
    identity = build_identity(name, split=split, reference_digest=digest, seed=seed, scale=scale)
    generated = generate_suite(suite, reference_name=reference_name, split=split, seed=seed, scale=scale,
                               workers=workers)
    # Novel alleles are named per donor, so each case is truthed with its own donor's.
    novel = {donor.subject_id: donor.novel for donor in generated.donors}
    truths = [truth_from_case(case, reference, novel=novel.get(case["groups"].get("subject")))
              for case in generated.cases]

    layout.suite_dir(name).mkdir(parents=True, exist_ok=True)
    write_cases(generated.cases, layout.cases(name))
    header = {"kind": TRUTH_KIND, "version": TRUTH_VERSION, "reference": reference_name,
              "reference_digest": digest, **genairr_provenance(),
              "suite": name, "split": split, "seed": seed, "scale": scale, "n_cases": len(generated.cases)}
    write_truth_jsonl(truths, layout.truth(name), header=header)
    write_fasta(generated.cases, layout.reads(name))
    if generated.donors:
        layout.donors(name).write_text(json.dumps(donors_to_json(generated.donors), indent=2) + "\n",
                                       encoding="utf-8")
    build = {
        "suite": name, "split": split, "seed": seed, "scale": scale,
        "reference": reference_name, "reference_digest": digest,
        "identity": identity, "config_hash": _digest(identity),
        "n_cases": len(generated.cases),
        "strata": generated.stats,
        "truth": truth_summary(truths),
    }
    if generated.donors:
        build["donors_summary"] = donors_summary(generated.donors, reference)
    build["workers"] = workers
    build["seconds"] = round(time.perf_counter() - start, 3)
    layout.build_json(name).write_text(json.dumps(build, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return build


def build_identity(suite: str, *, split: str, reference_digest: str, seed: int, scale: float) -> dict:
    """What a suite's cases and truth depend on, so a change to any of it makes its build stale: ``suite``,
    ``split``, ``reference_digest``, ``seed`` and ``scale``; ``suite_source`` (``suite_source_digest``) and
    ``truth_source`` (``truth_source_digest``), so an edited recipe, stratum or truth rule rebuilds without a
    version bump; ``truth_version`` (``TRUTH_VERSION``); and the GenAIRR in use (``GENAIRR_IDENTITY`` of
    ``genairr_provenance``: version, commit, dirty flag and the package's content digest)."""
    genairr = genairr_provenance()
    return {"suite": suite, "split": split, "reference_digest": reference_digest, "seed": int(seed),
            "scale": float(scale), "suite_source": suite_source_digest(suite), "truth_source": truth_source_digest(),
            "truth_version": TRUTH_VERSION, **{key: genairr[key] for key in GENAIRR_IDENTITY}}


def suite_source_files(suite: str) -> list[Path]:
    """The source files suite ``suite``'s cases are generated by: its catalog module (``suites.catalog_module``)
    and ``SUITE_SOURCES``."""
    return [Path(catalog_module(suite).__file__).resolve(), *(PACKAGE_DIR / name for name in SUITE_SOURCES)]


def suite_source_digest(suite: str) -> str:
    """``core.provenance.files_digest`` of ``suite_source_files(suite)``, relative to the package directory."""
    return files_digest(PACKAGE_DIR, suite_source_files(suite))


def truth_source_files() -> list[Path]:
    """The source files the truth is built by: every module of ``truth/`` and ``TRUTH_SOURCES``."""
    return [*sorted((PACKAGE_DIR / "truth").glob("*.py")), *(PACKAGE_DIR / name for name in TRUTH_SOURCES)]


def truth_source_digest() -> str:
    """``core.provenance.files_digest`` of ``truth_source_files()``, relative to the package directory."""
    return files_digest(PACKAGE_DIR, dict.fromkeys(truth_source_files()))


def build_changes(layout: RunLayout, suite: str, *, split: str, reference_digest: str, seed: int,
                  scale: float) -> list[str]:
    """Why the suite's build in ``layout`` is stale for this configuration: the ``build_identity`` fields its
    ``build.json`` records differently, ``"no build record"``, ``"identity (not recorded)"`` for a record
    written before identities were, or ``"truth version"`` when the truth file is not of the current
    ``TRUTH_VERSION``. Empty when the build is current."""
    try:
        record = json.loads(layout.build_json(suite).read_text(encoding="utf-8"))
        version = (read_truth_header(layout.truth(suite)) or {}).get("version")
    except (OSError, ValueError):
        return ["no build record"]
    recorded = record.get("identity")
    if not isinstance(recorded, dict):
        return ["identity (not recorded)"]
    wanted = build_identity(suite, split=split, reference_digest=reference_digest, seed=seed, scale=scale)
    changes = [key for key in sorted(recorded.keys() | wanted.keys()) if recorded.get(key) != wanted.get(key)]
    if not changes and record.get("config_hash") != _digest(wanted):
        changes = ["config_hash"]
    if version != TRUTH_VERSION:
        changes.append("truth version")
    return changes


def build_is_current(layout: RunLayout, suite: str, *, split: str, reference_digest: str, seed: int,
                     scale: float) -> bool:
    """True when the suite's ``build.json`` records this configuration's ``build_identity`` and its truth file the
    current truth version (``build_changes`` is empty): the build need not be redone, and runs on it may be
    reused."""
    return not build_changes(layout, suite, split=split, reference_digest=reference_digest, seed=seed, scale=scale)


def _digest(fields: dict) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
