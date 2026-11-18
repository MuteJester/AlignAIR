"""Self-checks of the stored runs, read only: the simulated run's oracle and staleness, and the real run's dev donors'
self-checks and truth health."""
from __future__ import annotations

from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import genairr_provenance, git_state
from alignair_assay.core.records import read_json
from alignair_assay.core.reference import Reference
from alignair_assay.real.fetch import PILOT_SPLIT
from alignair_assay.workflow.build import GENAIRR_IDENTITY, build_changes
from alignair_assay.workflow.orchestrate import read_run_config
from alignair_assay.workflow.summary import oracle_check
from alignair_assay.workflow.audit.base import _PACKAGE, check

# --- self-checks of the stored runs ---------------------------------------------------------------------------------

def sim_oracle(sim: RunLayout) -> dict:
    """``self_check.sim_oracle``: the oracle rule on the stored oracle scores of every suite of the ``all`` run
    (``workflow.summary.oracle_check``, read only), beside what its ``report.json`` recorded."""
    cfg = read_run_config(sim)
    recorded = {suite["name"]: suite.get("oracle") or {} for suite in (read_json(sim.report_json) or {}).get(
        "suites") or []}
    suites, bad, disagree = {}, [], []
    for name in cfg.suites:
        found = oracle_check(sim, name)
        suites[name] = {"status": found["status"], "cases": found["cases"], "problems": found["problems"],
                        "recorded": (recorded.get(name) or {}).get("status")}
        if found["status"] != "perfect":
            bad.append(name)
        if suites[name]["recorded"] != found["status"]:
            disagree.append(f"{name} (recorded {suites[name]['recorded']}, scores {found['status']})")
    cases = sum(entry["cases"] for entry in suites.values())
    summary = (f"oracle perfect on {len(suites) - len(bad)} of {len(suites)} suites ({cases} cases, "
               f"{cfg.split} split)" + (f"; the report disagrees on {len(disagree)}" if disagree else
                                         ", as its report recorded"))
    evidence = {"summary": summary, "run": str(sim.root), "split": cfg.split, "suites": suites}
    reasons = ([f"the oracle is not perfect on {', '.join(bad)}: its truth has a defect or is unchecked"] if bad
               else []) + ([f"the run's report recorded another oracle status than its scores give: "
                            f"{', '.join(disagree)}"] if disagree else [])
    if reasons:
        return check("self_check.sim_oracle", "fail", cases, evidence, "; ".join(reasons))
    return check("self_check.sim_oracle", "pass", cases, evidence)


def sim_staleness(sim: RunLayout) -> dict:
    """``self_check.sim_staleness`` (informational): the stored run's suite builds against this checkout's suite,
    truth and GenAIRR code (``workflow.build.build_changes``, which reads only), and its recorded assay commit and
    GenAIRR against the current ones; ``warn`` when a build is not current."""
    cfg = read_run_config(sim)
    manifest = read_json(sim.manifest) or {}
    digest = Reference.from_dataconfig(cfg.reference).digest()
    changes = {name: build_changes(sim, name, split=cfg.split, reference_digest=digest, seed=cfg.seed,
                                   scale=cfg.scale) for name in cfg.suites}
    stale = {name: found for name, found in changes.items() if found}
    current_genairr = genairr_provenance()
    evidence = {"run": str(sim.root), "split": cfg.split, "stale_builds": stale,
                "assay": {"recorded": manifest.get("git"), "current": git_state(_PACKAGE)},
                "genairr": {"recorded": {key: (manifest.get("genairr") or {}).get(key) for key in GENAIRR_IDENTITY},
                            "current": {key: current_genairr.get(key) for key in GENAIRR_IDENTITY}}}
    if stale:
        fields = sorted({field for found in stale.values() for field in found})
        evidence["summary"] = (f"{len(stale)} of {len(changes)} suite builds are not current for this checkout "
                               f"(changed: {', '.join(fields)})")
        return check("self_check.sim_staleness", "warn", len(changes), evidence,
                     "informational: the stored run's numbers were computed by other suite, truth or GenAIRR code "
                     "than this checkout's; the audit reads them as recorded and never re-scores the test split")
    evidence["summary"] = f"every one of the {len(changes)} suite builds is current for this checkout"
    return check("self_check.sim_staleness", "pass", len(changes), evidence)


def dev_donors(summary: dict) -> tuple[dict[str, dict], list[str]]:
    """The blocks of the pilot's dev donors (``real.fetch.PILOT_SPLIT``) in a ``real`` summary, in the pilot's order,
    and the dev donors the summary lacks: every donor check covers every dev donor, not only those a run holds."""
    blocks = summary.get("donors") or {}
    return ({donor: blocks[donor] for donor in PILOT_SPLIT["dev"] if donor in blocks},
            [donor for donor in PILOT_SPLIT["dev"] if donor not in blocks])


def _missing(donors: list[str]) -> list[str]:
    """The reason a donor check fails for dev donors the stored run lacks; none when it lacks none."""
    return [f"the dev donors {', '.join(donors)} are missing from the stored run"] if donors else []


def real_summary(real: RunLayout) -> dict:
    """The stored ``real`` run's summary; FileNotFoundError without one."""
    summary = read_json(real.real_summary_json())
    if summary is None:
        raise FileNotFoundError(f"no readable {real.real_summary_json()}: the run directory was not made by `real`")
    return summary


def real_oracle(real: RunLayout) -> dict:
    """``self_check.real_oracle``: every dev donor of the stored ``real`` run scored and its oracle self-check
    ``passed``, the pipeline path and the decoys included; a dev donor missing from the run fails."""
    donors, missing = dev_donors(real_summary(real))
    rows, bad = {}, []
    for donor, block in donors.items():
        found = block.get("oracle_self_check") or {}
        pipeline, decoys = found.get("pipeline") or {}, found.get("decoys") or {}
        rows[donor] = {"status": found.get("status") if block.get("status") == "done" else "stopped",
                       "scored": found.get("scored") or 0, "reads": found.get("reads"),
                       "ref_reads": found.get("ref_reads"), "pipeline_agrees": pipeline.get("agrees"),
                       "pipeline_reads": pipeline.get("reads"), "decoys_passed": decoys.get("passed")}
        if rows[donor]["status"] != "passed" or pipeline.get("passed") is not True or decoys.get("passed") is not True:
            bad.append(donor)
    scored = sum(row["scored"] for row in rows.values())
    evidence = {"summary": f"self-check passed on {len(rows) - len(bad)} of {len(PILOT_SPLIT['dev'])} dev donors "
                           f"({scored} oracle reads scored, pipeline path and decoys included)",
                "run": str(real.root), "donors": rows, "missing": missing}
    reasons = _missing(missing) + ([f"the truth self-check did not pass on {', '.join(bad)}"] if bad else [])
    if reasons:
        return check("self_check.real_oracle", "fail", scored, evidence, "; ".join(reasons))
    return check("self_check.real_oracle", "pass", scored, evidence)


def real_truth_health(real: RunLayout) -> dict:
    """``self_check.real_truth_health``: each dev donor's truth-health flags (carried genes whose reference-exact
    reads no donor allele explains, ``real.truth.truth_health``); ``warn`` when any is flagged or a donor has no
    truth-health block (never audited, so not clean); a dev donor missing from the run fails."""
    donors, missing = dev_donors(real_summary(real))
    flags, reads, unaudited = {}, 0, []
    for donor, block in donors.items():
        if not isinstance(block.get("truth_health"), dict):
            unaudited.append(donor)
            continue
        health = block["truth_health"]
        reads += health.get("reads") or 0
        genes = {entry["gene"]: entry for entry in health.get("genes") or []}
        for gene in health.get("flagged") or []:
            entry = genes.get(gene) or {}
            flags.setdefault(donor, []).append({"gene": gene, "reads": entry.get("reads"),
                                                "clonotypes": entry.get("clonotypes"),
                                                "alleles": entry.get("alleles"), "h0_only": entry.get("h0_only")})
    listed = "; ".join(f"{donor} {', '.join(flag['gene'] for flag in found)}" for donor, found in flags.items())
    evidence = {"summary": (f"{sum(map(len, flags.values()))} flags on {len(flags)} of {len(PILOT_SPLIT['dev'])} dev "
                            "donors" + (f": {listed}" if listed else "")
                            + (f"; no truth-health block for {', '.join(unaudited)}" if unaudited else "")),
                "flags": flags, "donors": list(donors), "unaudited": unaudited, "missing": missing}
    if missing:
        return check("self_check.real_truth_health", "fail", reads, evidence, "; ".join(_missing(missing)))
    warnings = ([f"{', '.join(unaudited)} have no truth-health block: their truth was never audited"] if unaudited
                else []) + (["carried genes whose reference-exact reads no donor allele explains: look at them before "
                             "trusting those genes' calls"] if flags else [])
    if warnings:
        return check("self_check.real_truth_health", "warn", reads, evidence,
                     "; ".join(warnings) + " (a warning, never a failure)")
    return check("self_check.real_truth_health", "pass", reads, evidence)
