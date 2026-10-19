"""Build one suite into a run directory: generate its cases, build their truth, write reads and the build record.

The files are ``core.layout.RunLayout``'s ``cases``, ``truth``, ``reads`` and ``build_json`` of the suite;
``build.json`` is removed first and written last, so it exists only for a complete build.
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
from alignair_assay.suites.engine import check_split, generate_suite
from alignair_assay.truth.case import truth_from_case
from alignair_assay.truth.io import TRUTH_KIND, TRUTH_VERSION, write_truth_jsonl
from alignair_assay.truth.summary import truth_summary


def build_suite(name: str, layout: RunLayout, *, reference_name: str, split: str, seed: int, scale: float,
                workers: int) -> dict:
    """Generate suite ``name`` (``suites.engine.generate_suite``) and write its files; returns the build record.

    The truth header holds ``kind``, ``version``, ``reference``, ``reference_digest``, the GenAIRR provenance,
    ``suite``, ``split``, ``seed``, ``scale`` and ``n_cases``. The build record (``build.json``) holds the same
    configuration, its ``config_hash``, the per-stratum generation stats (``strata``), the truth summary
    (``truth``: excluded cases, warnings, unscorable genes, simulator disagreements), ``workers`` and the wall
    time in ``seconds``. Cases and truth do not depend on ``workers``.
    """
    start = time.perf_counter()
    suite = get_suite(name)
    check_split(split)
    seed, scale = int(seed), float(scale)
    layout.build_json(name).unlink(missing_ok=True)
    reference = Reference.from_dataconfig(reference_name)
    digest = reference.digest()
    generated = generate_suite(suite, reference_name=reference_name, split=split, seed=seed, scale=scale,
                               workers=workers)
    truths = [truth_from_case(case, reference) for case in generated.cases]

    layout.suite_dir(name).mkdir(parents=True, exist_ok=True)
    write_cases(generated.cases, layout.cases(name))
    header = {"kind": TRUTH_KIND, "version": TRUTH_VERSION, "reference": reference_name,
              "reference_digest": digest, **genairr_provenance(),
              "suite": name, "split": split, "seed": seed, "scale": scale, "n_cases": len(generated.cases)}
    write_truth_jsonl(truths, layout.truth(name), header=header)
    write_fasta(generated.cases, layout.reads(name))
    build = {
        "suite": name, "split": split, "seed": seed, "scale": scale,
        "reference": reference_name, "reference_digest": digest,
        "config_hash": config_hash(name, split=split, reference_digest=digest, seed=seed, scale=scale),
        "n_cases": len(generated.cases),
        "strata": generated.stats,
        "truth": truth_summary(truths),
        "workers": workers,
        "seconds": round(time.perf_counter() - start, 3),
    }
    layout.build_json(name).write_text(json.dumps(build, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return build


def config_hash(suite: str, *, split: str, reference_digest: str, seed: int, scale: float) -> str:
    """Hex SHA-256 of the canonical JSON (sorted keys, no spaces) of the configuration a suite's cases and truth
    depend on: ``{suite, split, reference_digest, seed, scale}``."""
    config = {"suite": suite, "split": split, "reference_digest": reference_digest, "seed": seed, "scale": scale}
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
