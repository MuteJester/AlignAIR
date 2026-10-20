"""Build one suite into a run directory: generate its cases, build their truth, write reads and the build record.

The files are ``core.layout.RunLayout``'s ``cases``, ``truth``, ``reads``, ``donors`` (suites with donors only)
and ``build_json`` of the suite; ``build.json`` is removed first and written last, so it exists only for a
complete build.
"""
from __future__ import annotations

import hashlib
import json
import time

from alignair_assay.core.cases import write_cases, write_fasta
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import genairr_provenance
from alignair_assay.core.reference import Reference
from alignair_assay.suites import get_suite
from alignair_assay.suites.donors import donors_summary, donors_to_json
from alignair_assay.suites.engine import check_split, generate_suite
from alignair_assay.truth.case import truth_from_case
from alignair_assay.truth.io import TRUTH_KIND, TRUTH_VERSION, read_truth_header, write_truth_jsonl
from alignair_assay.truth.summary import truth_summary


def build_suite(name: str, layout: RunLayout, *, reference_name: str, split: str, seed: int, scale: float,
                workers: int) -> dict:
    """Generate suite ``name`` (``suites.engine.generate_suite``) and write its files; returns the build record.

    The truth header holds ``kind``, ``version``, ``reference``, ``reference_digest``, the GenAIRR provenance,
    ``suite``, ``split``, ``seed``, ``scale`` and ``n_cases``. The build record (``build.json``) holds the same
    configuration, its ``config_hash``, the per-stratum generation stats (``strata``), the truth summary
    (``truth``: excluded cases, warnings, unscorable genes, simulator calls, primary fallbacks),
    ``workers`` and the wall time in ``seconds``. A suite with donors also gets ``donors.json``
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
        "config_hash": config_hash(name, split=split, reference_digest=digest, seed=seed, scale=scale),
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


def build_is_current(layout: RunLayout, suite: str, *, split: str, reference_digest: str, seed: int,
                     scale: float) -> bool:
    """True when the suite's ``build.json`` has the ``config_hash`` of this configuration and its truth file the
    current truth version (``TRUTH_VERSION``): the build need not be redone, and runs on it may be reused."""
    try:
        record = json.loads(layout.build_json(suite).read_text(encoding="utf-8"))
        version = (read_truth_header(layout.truth(suite)) or {}).get("version")
    except (OSError, ValueError):
        return False
    wanted = config_hash(suite, split=split, reference_digest=reference_digest, seed=int(seed), scale=float(scale))
    return record.get("config_hash") == wanted and version == TRUTH_VERSION


def config_hash(suite: str, *, split: str, reference_digest: str, seed: int, scale: float) -> str:
    """Hex SHA-256 of the canonical JSON (sorted keys, no spaces) of the configuration a suite's cases and truth
    depend on: ``{suite, split, reference_digest, seed, scale}``."""
    config = {"suite": suite, "split": split, "reference_digest": reference_digest, "seed": seed, "scale": scale}
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
