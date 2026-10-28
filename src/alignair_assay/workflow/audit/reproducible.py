"""Reproducibility: a dev build made twice, and one real donor's truth and scores recomputed from the stored inputs,
compared with what is stored."""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

from alignair_assay.core.jsonl import read_header, read_jsonl
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import genairr_provenance, sha256_file, tree_digest
from alignair_assay.core.records import read_json
from alignair_assay.real.fetch import PILOT_SPLIT
from alignair_assay.real.genotype import genotype_from_json, merge_identical
from alignair_assay.workflow import real_score, real_steps
from alignair_assay.workflow.build import GENAIRR_IDENTITY, build_suite
from alignair_assay.workflow.real import donor_scoring, scoring_inputs, self_check
from alignair_assay.workflow.summary import ORACLE
from alignair_assay.workflow.audit.base import AuditConfig, CODE_FIELDS, REPRO_FILES, check
from alignair_assay.workflow.audit.stored import real_summary

# --- reproducibility ----------------------------------------------------------------------------------------------

def genairr_digest() -> str | None:
    """The content digest of the GenAIRR source in use, read from its files now (``core.provenance.tree_digest``;
    ``genairr_provenance`` computes it once per process)."""
    source = genairr_provenance().get("genairr_source")
    return tree_digest(source) if source else None


def genairr_identity() -> dict:
    """The GenAIRR in use as a build records it (``workflow.build.GENAIRR_IDENTITY``: version, commit, dirty flag),
    its source digest read now (``genairr_digest``): a commit or an edit changes it."""
    found = genairr_provenance()
    return {**{key: found.get(key) for key in GENAIRR_IDENTITY}, "genairr_source_sha256": genairr_digest()}


def sim_reproducible(cfg: AuditConfig, first: RunLayout, second: RunLayout) -> dict:
    """``reproducible.sim_build``: ``cfg.repro_suite`` built twice on the dev split at ``cfg.repro_scale`` (in
    ``first`` and ``second``); its cases, truth and reads (``REPRO_FILES``, required in both builds) and donors (when
    either build has them) must be byte-identical. ``skipped (GenAIRR
    moved)`` when GenAIRR's source changed before, between or after the builds, or the builds record another
    GenAIRR (the user edits it live): then nothing is compared."""
    name, suite = "reproducible.sim_build", cfg.repro_suite
    digests = [genairr_digest()]
    builds = []
    for layout in (first, second):
        builds.append(build_suite(suite, layout, reference_name=cfg.reference, split="dev", seed=cfg.seed,
                                  scale=cfg.repro_scale, workers=cfg.workers))
        digests.append(genairr_digest())
    recorded = [{key: build["identity"].get(key) for key in GENAIRR_IDENTITY} for build in builds]
    evidence = {"suite": suite, "scale": cfg.repro_scale, "genairr_digests": digests, "genairr_recorded": recorded}
    if len(set(digests)) > 1 or recorded[0] != recorded[1]:
        evidence["summary"] = "GenAIRR's source changed during the two builds; nothing compared"
        return check(name, "skipped", 0, evidence, "skipped (GenAIRR moved): GenAIRR's source changed during the "
                                                   "builds, so they need not match")
    files = {"cases": RunLayout.cases, "truth": RunLayout.truth, "reads": RunLayout.reads, "donors": RunLayout.donors}
    compared, missing = {}, []
    for label, path in files.items():
        a, b = path(first, suite), path(second, suite)
        if label in REPRO_FILES:
            missing += [f"{label} ({which})" for which, found in (("first", a), ("second", b)) if not found.is_file()]
        if not a.is_file() and not b.is_file():
            continue
        compared[label] = {"first": sha256_file(a) if a.is_file() else None,
                           "second": sha256_file(b) if b.is_file() else None}
    differ = [label for label, found in compared.items() if found["first"] != found["second"]]
    n = builds[0]["n_cases"]
    evidence.update(files=compared, missing=missing, identity_same=builds[0]["identity"] == builds[1]["identity"],
                    summary=(f"{suite} at scale {cfg.repro_scale} built twice ({n} cases): "
                             + (f"{', '.join(missing)} missing" if missing else
                                f"{', '.join(differ)} differ" if differ else
                                f"{', '.join(compared)} byte-identical")))
    if missing:
        return check(name, "fail", n, evidence, f"the builds of {suite} are missing {', '.join(missing)}: "
                                                f"{', '.join(REPRO_FILES)} are compared in both builds")
    if differ or n != builds[1]["n_cases"]:
        return check(name, "fail", n, evidence, f"two builds of {suite} on one GenAIRR differ in {', '.join(differ)}: "
                                                "the suite generator or its truth is not deterministic")
    return check(name, "pass", n, evidence)


def jsonl_content(path: Path, aliases: Mapping[str, str] | None = None) -> dict:
    """What a JSON-lines record file holds, by digest: ``file`` (its bytes' SHA-256), ``header`` (the header line's
    record without ``CODE_FIELDS``, each string value found in ``aliases`` replaced by its alias, digested; None
    without a header), ``code`` (the header's ``code_sha256``: the fingerprint also hashes the inputs, which the
    header compares itself), ``rows`` (the other lines as ``core.jsonl`` reads them, digested) and their count
    ``n``: whitespace and blank lines are not content."""
    header, rows = read_header(path), read_jsonl(path)
    content = None if header is None else _aliased(
        {key: value for key, value in header.items() if key not in CODE_FIELDS}, aliases or {})
    return {"file": sha256_file(path), "header": None if content is None else _content_digest(content),
            "code": None if header is None else header.get("code_sha256"),
            "rows": _content_digest(rows), "n": len(rows)}


def _content_digest(value) -> str:
    """The SHA-256 of ``value`` as canonical JSON (keys sorted)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _aliased(value, aliases: Mapping[str, str]):
    """``value`` with every string that is a key of ``aliases`` replaced by its alias, in nested dicts and lists."""
    if isinstance(value, dict):
        return {key: _aliased(item, aliases) for key, item in value.items()}
    if isinstance(value, list):
        return [_aliased(item, aliases) for item in value]
    return aliases.get(value, value) if isinstance(value, str) else value


def compare_records(stored: Path, recomputed: Path, *, aliases: Mapping[str, str] | None = None) -> dict:
    """``stored`` against ``recomputed`` (``jsonl_content``): ``identical`` (the content: the rows and the header
    without its code fields), ``file_identical`` (the bytes), ``code_moved`` (the code fields differ), the ``rows``
    count, and ``differs``: what differs (``rows``, ``header``). ``aliases`` maps the byte digest of a stored input
    file to its recomputed twin's, when the two were found to hold the same content: a stored header that names the
    stored file's digest then matches a recomputed one naming its twin's (a file written before JSON-lines gzip was
    reproducible carries a time in its header, so its bytes differ while its content does not)."""
    a, b = jsonl_content(stored, aliases), jsonl_content(recomputed)
    differs = [part for part in ("rows", "header") if a[part] != b[part]]
    return {"identical": not differs, "file_identical": a["file"] == b["file"], "code_moved": a["code"] != b["code"],
            "rows": b["n"], "differs": differs}


def real_reproducible(cfg: AuditConfig, real: RunLayout, scratch: RunLayout) -> dict:
    """``reproducible.real_scores``: ``cfg.real_donor``'s truth (``real_steps.step_truth``), its self-check (the
    pipeline oracle's AIRR TSV and scores included) and every stored run's scores (``real_score.step_scores``)
    recomputed in ``scratch`` from the stored reads, ``reads.tsv``, genotype and predictions, each compared with the
    stored one (``compare_records``; the oracle's TSV byte for byte, the self-check as data). The IUIS map is read from
    the donor's recorded OGRDB file under the data root; the SD3 genes from the stored summary."""
    name, donor = "reproducible.real_scores", cfg.real_donor
    summary = real_summary(real)
    block = (summary.get("donors") or {}).get(donor)
    if block is None or block.get("status") != "done" or donor not in PILOT_SPLIT["dev"]:
        return check(name, "fail", 0, {"summary": f"{donor} is not a scored dev donor of the stored run"},
                     f"{donor} is not a scored dev donor of {real.root}")
    for path in (RunLayout.real_reads, RunLayout.real_reads_meta, RunLayout.real_genotype_json):
        _copy(path(real, donor), path(scratch, donor))
    reference = real_steps.load_reference(summary["config"]["reference"])
    fetched = read_json(real.real_fetch_json(donor)) or {}
    ogrdb = ((fetched.get("records") or {}).get("genotypes") or {}).get("ogrdb") or {}
    iuis = real_steps.load_iuis(Path(cfg.data_root) / ogrdb["path"] if ogrdb.get("path") else None, reference)
    genotype = genotype_from_json((read_json(scratch.real_genotype_json(donor)) or {})["genotype"])
    truths, _, _ = real_steps.step_truth(donor, scratch, genotype, reference, iuis)
    files = {"truth": compare_records(real.real_truth(donor), scratch.real_truth(donor))}
    # the scores' inputs name the truth file by its bytes: the stored truth's digest stands for the recomputed one's
    # when the two hold the same content
    aliases = ({sha256_file(real.real_truth(donor)): sha256_file(scratch.real_truth(donor))}
               if files["truth"]["identical"] else {})
    data = donor_scoring(donor, scratch, truths, genotype, iuis, reference, merge_identical(reference),
                         (summary.get("sensitivity") or {}).get("genes") or ())
    common = scoring_inputs(donor, scratch, iuis, reference)
    redone = self_check(data, scratch, common)
    files[f"shared/{ORACLE}"] = compare_records(real.real_scores(donor, "shared", ORACLE),
                                                scratch.real_scores(donor, "shared", ORACLE), aliases=aliases)
    oracle_tsv = [sha256_file(layout.real_predictions(ORACLE, donor, "shared")) for layout in (real, scratch)]
    self_same = _comparable(redone) == _comparable(block.get("oracle_self_check") or {})
    missing, runs_compared = [], 0
    for arm, runs in (block.get("runs") or {}).items():
        for tool, run in runs.items():
            if run.get("status") != "ok":
                continue
            stored = real.real_scores(donor, arm, tool)
            if not stored.is_file():
                missing.append(f"{arm}/{tool}")  # an ok run is scored; a stored run without its scores is broken
                continue
            _copy(real.real_predictions(tool, donor, arm), scratch.real_predictions(tool, donor, arm))
            recorded = ((read_json(real.real_run_json(tool, donor, arm)) or {}).get("provenance") or {}).get(
                "predictions_sha256")
            real_score.step_scores(data, tool, arm, scratch, {**common, "predictions_sha256": recorded})
            files[f"{arm}/{tool}"] = compare_records(stored, scratch.real_scores(donor, arm, tool), aliases=aliases)
            runs_compared += 1
    differ = [label for label, found in files.items() if not found["identical"]]
    differ += [] if oracle_tsv[0] == oracle_tsv[1] else ["oracle TSV"]
    differ += [] if self_same else ["self-check"]
    differ += [f"{label} (its stored scores are missing)" for label in missing]
    differ += [] if runs_compared else ["no run's scores (no ok run was compared)"]
    moved = [label for label, found in files.items() if found["code_moved"]]
    evidence = {"donor": donor, "files": files, "oracle_tsv_identical": oracle_tsv[0] == oracle_tsv[1],
                "self_check_identical": self_same, "code_moved": moved, "runs_compared": runs_compared,
                "missing_scores": missing,
                "summary": (f"{donor}: truth of {len(truths)} reads, its self-check and {runs_compared} runs' scores "
                            "recomputed: " + (f"{', '.join(differ)} differ" if differ else
                                              "every one identical in content")
                            + (f" (the code computing {', '.join(moved)} moved since: compared without its digest)"
                               if moved else ""))}
    if differ:
        return check(name, "fail", len(truths), evidence,
                     f"recomputed from {donor}'s stored inputs, {', '.join(differ)} differ from the stored records: "
                     "the real truth or scoring is not reproducible, or the stored run is stale for this code")
    return check(name, "pass", len(truths), evidence)


def _comparable(found: dict) -> dict:
    """A self-check as JSON reads it back (tuples as lists, keys as strings)."""
    return json.loads(json.dumps(found, sort_keys=True))


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
