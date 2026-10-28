"""Null comparisons (one checkpoint under two labels on ``all``, ``diagnose`` and ``real``), the positive control (a
perturbed checkpoint must get worse) and the cross-surface agreement of ``all`` and ``diagnose``."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path

from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import sha256_file
from alignair_assay.core.records import read_json
from alignair_assay.core.reference import Reference
from alignair_assay.runners.base import read_run_json
from alignair_assay.scoring.pipeline import read_scores, truth_reference
from alignair_assay.suites import get_suite
from alignair_assay.workflow.build import build_changes, build_suite
from alignair_assay.workflow.orchestrate import RunConfig, run_all
from alignair_assay.workflow.real import CHECKPOINT_PAIR, RealConfig, run_real
from alignair_assay.workflow.summary import ORACLE
from alignair_assay.workflow.audit.base import (
    AGREEMENT_TOOL,
    AuditConfig,
    HEAD_BOUNDARY_PREFIXES,
    NULL_LABELS,
    POSITIVE_METRIC,
    POSITIVE_MODULE,
    REAL_HEADLINE,
    SIM_HEADLINE,
    _EXAMPLES,
    _TIE,
    _percent,
    _signed,
    check,
    verdict,
)
from alignair_assay.workflow.audit.reproducible import genairr_identity

# --- null comparisons ---------------------------------------------------------------------------------------------

def compare_problems(comparison: Mapping[str, dict], where: str) -> list[str]:
    """What makes a ``stats.compare`` result (metric -> entry) of one model against itself not null: a Δ other than 0
    (None means no pair; ``pairing`` holds a comparison to its pairs), a verdict other than a tie, a read won or lost,
    each side's own scores differing, or a read one side scores left unpaired (``n_paired`` below either side's own
    ``n``)."""
    problems = []
    for metric, entry in comparison.items():
        delta = entry.get("delta")
        if delta not in (None, 0):
            problems.append(f"{where} {metric}: Δ {delta:+.6g}")
        if entry.get("verdict") not in (None, _TIE):
            problems.append(f"{where} {metric}: verdict {entry['verdict']}")
        if entry.get("wins") or entry.get("losses"):
            problems.append(f"{where} {metric}: {entry.get('wins')} reads won, {entry.get('losses')} lost")
        if entry.get("a") != entry.get("b"):
            problems.append(f"{where} {metric}: each side's own scores differ")
        sides = [(entry.get(side) or {}).get("n") or 0 for side in ("a", "b")]
        if any((entry.get("n_paired") or 0) != n for n in sides):
            problems.append(f"{where} {metric}: pairs {entry.get('n_paired') or 0} of the {sides[0]} and {sides[1]} "
                            "reads the two sides score")
    return problems


def pairing(comparison: Mapping[str, dict], headline: Iterable[str], where: str, *,
            offered: int | None = None) -> tuple[int, list[str]]:
    """What a null comparison rests on, and why it may rest on too little: ``n``, the fewest reads any of the
    ``headline`` metrics pairs (a headline metric not compared pairs none); problems when no metric pairs a read, a
    headline metric was not compared (what is absent is never null), a headline metric pairs no read, or, given the
    reads ``offered`` (the scored cases), the most any headline metric pairs falls short of them (a read never reached
    the comparison). A headline metric that applies to fewer reads (D top-1 where no D is observable) pairs fewer;
    ``compare_problems`` already holds each metric to every read either side scores."""
    problems = [] if _paired(comparison) else [f"{where}: no metric pairs a read"]
    counts = {metric: (comparison[metric].get("n_paired") or 0) if metric in comparison else None
              for metric in headline}
    problems += [f"{where} {metric}: a headline metric, not compared" for metric, n in counts.items() if n is None]
    problems += [f"{where} {metric}: pairs no read" for metric, n in counts.items() if n == 0]
    paired = [n for n in counts.values() if n is not None]
    if offered is not None and paired and max(paired) != offered:
        problems.append(f"{where}: its headline metrics pair at most {max(paired)} of the {offered} reads offered")
    return min((n or 0 for n in counts.values()), default=0), problems


def changes_problems(changes: dict | None, where: str) -> list[str]:
    """What makes ``stats.read_changes`` of one model against itself not null: a read fixed or broken (overall or on a
    metric), or a read only one side scores."""
    if not changes:
        return [f"{where}: no reads fixed and broken were counted"]
    problems = []
    reads = changes.get("reads") or {}
    if reads.get("fixed") or reads.get("broken"):
        problems.append(f"{where}: {reads.get('fixed')} reads fixed, {reads.get('broken')} broken")
    for metric, found in (changes.get("metrics") or {}).items():
        if found.get("fixed") or found.get("broken"):
            problems.append(f"{where} {metric}: {found['fixed']} fixed, {found['broken']} broken")
    if changes.get("a_only") or changes.get("b_only"):
        problems.append(f"{where}: {changes.get('a_only')} reads scored by a only, {changes.get('b_only')} by b only")
    return problems


def _paired(comparison: Mapping[str, dict]) -> int:
    return sum(1 for entry in comparison.values() if entry.get("n_paired"))


def _null_check(name: str, problems: list[str], n: int, evidence: dict, what: str) -> dict:
    evidence["problems"] = problems[:_EXAMPLES]
    evidence["problem_count"] = len(problems)
    if problems:
        return check(name, "fail", n, evidence, f"{what} is not null ({len(problems)} problems, e.g. "
                                                f"{'; '.join(problems[:3])}): the pipeline is non-deterministic or "
                                                "pairs reads unequally")
    return check(name, "pass", n, evidence)


def all_null(cfg: AuditConfig, scratch: RunLayout) -> list[dict]:
    """``null.all`` and ``self_check.sim_oracle_current``: ``all`` with the checkpoint as ``alignair-local@a`` and
    ``@b`` (``workflow.orchestrate.run_all``) on fresh dev builds of ``cfg.null_suites`` at ``cfg.null_scale`` in
    ``scratch``; every comparison must be null (``compare_problems``), and the oracle perfect on every suite.

    ``all`` scores a run only on a build current for the GenAIRR in use, so GenAIRR edited or committed while it ran
    (``genairr_identity`` before and after) leaves its runs stale and unscored: it is run again once (rebuilding what
    moved), and both checks are ``skipped (GenAIRR moved)`` when GenAIRR moves again."""
    tools = tuple(f"alignair-local@{label}" for label in NULL_LABELS)
    run_cfg = RunConfig(suites=tuple(cfg.null_suites), tools=tools, split="dev", reference=cfg.reference,
                        seed=cfg.seed, scale=cfg.null_scale, workers=cfg.workers, threads=cfg.threads,
                        device=cfg.device, checkpoints={label: cfg.checkpoint for label in NULL_LABELS})
    moved = 0
    while True:
        before = genairr_identity()
        returned = run_all(run_cfg, scratch)
        if genairr_identity() == before:
            break
        moved += 1
        if moved > 1:
            evidence = {"summary": "GenAIRR changed while all ran, twice; nothing compared", "genairr_moved": moved}
            reason = "skipped (GenAIRR moved): GenAIRR changed while all ran, so its builds were stale when scored"
            return [check("null.all", "skipped", 0, evidence, reason),
                    check("self_check.sim_oracle_current", "skipped", 0, dict(evidence), reason)]
    report = read_json(scratch.report_json) or {}
    problems = [f"{run['tool']} on {run['suite']}/{run['arm']} {run['status']}: {run.get('error')}"
                for run in report.get("failed_runs") or []]
    built = {suite["name"]: suite for suite in report.get("suites") or []}
    suites, reads = {}, 0
    for name in cfg.null_suites:
        offered = ((built.get(name) or {}).get("cases") or 0) - ((built.get(name) or {}).get("excluded") or 0)
        for arm in get_suite(name).arms:
            found = read_json(scratch.comparison_json(name, arm))
            if found is None:
                problems.append(f"{name}/{arm}: not compared")
                continue
            rows = found.get("comparison") or {}
            problems += compare_problems(rows, f"{name}/{arm}")
            problems += changes_problems(found.get("changes"), f"{name}/{arm}")
            n, short = pairing(rows, SIM_HEADLINE, f"{name}/{arm}", offered=offered)
            problems += short
            problems += [f"{name}/{arm} {metric}: the informational summaries differ"
                         for metric, sides in (found.get("informational") or {}).items()
                         if sides.get("a") != sides.get("b")]
            slices = found.get("by_slice") or {}
            if slices.get("a") != slices.get("b"):
                problems.append(f"{name}/{arm}: the two tools' per-slice summaries differ")
            pair = [scratch.predictions(tool, name, arm) for tool in tools]
            suites[f"{name}/{arm}"] = {"metrics": len(rows), "paired_metrics": _paired(rows), "n": n,
                                       "offered": offered,
                                       "predictions_identical": all(path.is_file() for path in pair)
                                       and sha256_file(pair[0]) == sha256_file(pair[1])}
            reads += n if arm == "shared" else 0
    runs = {}
    for tool, suite, arm in dict.fromkeys([(ORACLE, name, "shared") for name in cfg.null_suites]
                                          + [(tool, name, arm) for tool in tools for name in cfg.null_suites
                                             for arm in get_suite(name).arms]):
        try:
            runs[f"{tool} {suite}/{arm}"] = read_run_json(scratch.run_json(tool, suite, arm)).returncode
        except (OSError, ValueError, TypeError):
            runs[f"{tool} {suite}/{arm}"] = None
    problems += [f"{run}: {'no run record' if code is None else f'return code {code}'}"
                 for run, code in runs.items() if code != 0]
    evidence = {"summary": ", ".join(f"{key}: {entry['metrics']} metrics, every headline metric pairing at least "
                                     f"{entry['n']} of {entry['offered']} reads, predictions "
                                     f"{'byte-identical' if entry['predictions_identical'] else 'differ'}"
                                     for key, entry in suites.items()) or "nothing compared",
                "tools": list(tools), "scale": cfg.null_scale, "returncode": returned, "suites": suites,
                "runs": runs, "genairr_moved": moved}
    null = _null_check("null.all", problems, reads, evidence, "all's comparison of one checkpoint with itself")
    oracle = {suite["name"]: suite.get("oracle") or {} for suite in report.get("suites") or []}
    bad = [name for name in cfg.null_suites if (oracle.get(name) or {}).get("status") != "perfect"]
    cases = sum((oracle.get(name) or {}).get("cases") or 0 for name in cfg.null_suites)
    current = {"summary": f"oracle perfect on {len(cfg.null_suites) - len(bad)} of {len(cfg.null_suites)} fresh dev "
                          f"suites ({cases} cases) at this checkout's code",
               "suites": {name: oracle.get(name) for name in cfg.null_suites}}
    if bad:
        return [null, check("self_check.sim_oracle_current", "fail", cases, current,
                            f"the oracle is not perfect on the fresh dev {', '.join(bad)}")]
    return [null, check("self_check.sim_oracle_current", "pass", cases, current)]


def diagnose_null(cfg: AuditConfig, scratch: RunLayout) -> dict:
    """``null.diagnose``: dev builds of ``cfg.diagnose_suites`` at ``cfg.null_scale`` in ``scratch`` (the null
    ``all`` run's directory: a suite it built current is not built again, ``ensure_built``), then the checkpoint file
    diagnosed against itself loaded in memory (``workflow.diagnose.evaluate(..., against=(model,
    reference_set))``): two model keys, each diagnosed on the same reads, whose comparison must be null. One model
    key (one record compared with itself, zero by construction) fails, and so does a comparison missing anything it
    must hold: a suite of ``cfg.diagnose_suites`` (or one more), a suite's head family or a boundary metric of each
    gene and edge in it, its end-to-end family or a headline metric in it, or the suite's scored reads
    (``reads.scored``, what the headline metrics must pair)."""
    # torch and alignair are imported only when diagnostics run
    from alignair_assay.diagnostics.ablation import DEFAULT_VARIANTS
    from alignair_assay.diagnostics.replay import load_model_checked
    from alignair_assay.workflow.diagnose import evaluate, parse_device

    names = list(cfg.diagnose_suites)
    ensure_built(cfg, scratch, names)
    device = parse_device(cfg.device, fix="--device cpu")
    in_memory = load_model_checked(cfg.checkpoint, truth_reference(scratch, names), device=device)
    variants = DEFAULT_VARIANTS if cfg.diagnose_variants is None else cfg.diagnose_variants
    result = evaluate(cfg.checkpoint, scratch, names, device=device, batch_size=cfg.batch_size, variants=variants,
                      profile=False, against=in_memory)
    against = result.against
    problems = []
    if against["same_model"] or against["baseline"] == result.model_key:
        problems.append("one model key: the file was compared with its own record, zero by construction")
    compared = list(against.get("suites") or {})
    if sorted(compared) != sorted(names):
        problems.append(f"the comparison holds the suites {', '.join(compared) or 'none'}, not the diagnosed "
                        f"{', '.join(names)}")
    suites, reads = {}, 0
    for name, suite in (against.get("suites") or {}).items():
        offered = (((getattr(result, "suites", None) or {}).get(name) or {}).get("reads") or {}).get("scored")
        if offered is None:
            problems.append(f"{name}: the diagnosis records no scored reads (reads.scored)")
        entry = {"cases": suite.get("cases"), "offered": offered}
        for family in ("heads", "baseline"):
            found = suite.get(family)
            if not found:
                entry[family] = None
                continue
            where = f"{name} {family}"
            problems += compare_problems(found["comparison"], where)
            problems += changes_problems(found.get("changes"), where)
            entry[family] = {"metrics": len(found["comparison"]), "paired_metrics": _paired(found["comparison"]),
                             "fixed": found["changes"]["reads"]["fixed"], "broken": found["changes"]["reads"]["broken"]}
            if family == "baseline":
                entry[family]["n"], short = pairing(found["comparison"], SIM_HEADLINE, where, offered=offered)
                problems += short
            else:
                problems += [] if entry[family]["paired_metrics"] else [f"{where}: no metric pairs a read"]
                problems += [f"{where}: no {prefix}* boundary metric" for prefix in HEAD_BOUNDARY_PREFIXES
                             if not any(metric.startswith(prefix) for metric in found["comparison"])]
                problems += [f"{where} {metric}: pairs no read" for metric, row in found["comparison"].items()
                             if metric.startswith(HEAD_BOUNDARY_PREFIXES) and not row.get("n_paired")]
        if entry["heads"] is None:
            problems.append(f"{name}: no head family was compared")
        if entry["baseline"] is None:
            problems.append(f"{name}: no end-to-end family was compared (no variant ran)")
        reads += (entry["baseline"] or {}).get("n") or 0
        suites[name] = entry
    shown = []
    for name, entry in suites.items():
        families = [f"{family} {found['metrics']} metrics" for family, found in entry.items()
                    if isinstance(found, dict)]
        least = (entry["baseline"] or {}).get("n")
        shown.append(f"{name} {entry['cases']} reads ({', '.join(families)}; headline metrics pair at least {least})")
    evidence = {"summary": f"file {result.model_key} against in-memory {against['baseline']}: {', '.join(shown)}",
                "model_key": result.model_key, "baseline": against["baseline"], "device": device,
                "batch_size": cfg.batch_size, "variants": list(variants), "suites": suites}
    return _null_check("null.diagnose", problems, reads, evidence,
                       "diagnose's comparison of the checkpoint with itself in memory")


def ensure_built(cfg: AuditConfig, layout: RunLayout, names: Iterable[str]) -> list[str]:
    """Build each suite of ``names`` on the dev split at ``cfg.null_scale`` in ``layout`` unless its build there is
    current (``workflow.build.build_changes``): the null surfaces share one build of a suite, never two. Returns the
    suites built."""
    digest = Reference.from_dataconfig(cfg.reference).digest()
    built = []
    for name in names:
        if build_changes(layout, name, split="dev", reference_digest=digest, seed=cfg.seed, scale=cfg.null_scale):
            build_suite(name, layout, reference_name=cfg.reference, split="dev", seed=cfg.seed, scale=cfg.null_scale,
                        workers=cfg.workers)
            built.append(name)
    return built


def shared_suite(cfg: AuditConfig) -> str | None:
    """The first suite of ``cfg.null_suites`` that ``diagnose`` diagnoses too (S1): the build ``all`` and
    ``diagnose`` share, which the positive control and the agreement check use; None when there is none."""
    return next((name for name in cfg.null_suites if name in cfg.diagnose_suites), None)


# --- the positive control and cross-surface agreement ---------------------------------------------------------------

def positive_diagnose(cfg: AuditConfig, layout: RunLayout, scratch: RunLayout) -> dict:
    """``positive.diagnose``: the checkpoint with ``POSITIVE_MODULE`` zeroed (``diagnostics.perturb.zero_module``,
    written to ``scratch``) diagnosed against the checkpoint (``workflow.diagnose.evaluate``, the baseline: its
    records from ``null.diagnose`` are reused) on the shared suite's build in ``layout`` (``shared_suite``), with the
    null's settings. The perturbed model is the model under test, so the direction is known: ``POSITIVE_METRIC`` must
    be compared on some reads with Δ < 0 and the verdict ``a_better`` (baseline better), reads must be fixed or
    broken, more broken than fixed, the summary must carry that Δ and verdict, and the suite detail
    (``layout.diagnostics_against_md``) must render the metric's verdict as "baseline better". ``n`` is the reads the
    metric pairs."""
    # torch and alignair are imported only when diagnostics run
    from alignair_assay.diagnostics.ablation import DEFAULT_VARIANTS
    from alignair_assay.diagnostics.perturb import zero_module
    from alignair_assay.report.diagnostics_against import AGAINST_VERDICTS
    from alignair_assay.workflow.diagnose import evaluate, parse_device

    name, suite = "positive.diagnose", shared_suite(cfg)
    if suite is None:
        return check(name, "fail", 0, {"summary": "no suite both null surfaces build"},
                     "the null all and diagnose suites share no suite to perturb on")
    perturbed = Path(scratch.root) / "perturbed.alignair"
    zeroed = zero_module(cfg.checkpoint, perturbed, POSITIVE_MODULE)
    ensure_built(cfg, layout, [suite])
    device = parse_device(cfg.device, fix="--device cpu")
    variants = DEFAULT_VARIANTS if cfg.diagnose_variants is None else cfg.diagnose_variants
    result = evaluate(str(perturbed), layout, [suite], device=device, batch_size=cfg.batch_size, variants=variants,
                      profile=False, against=cfg.checkpoint)
    against = result.against
    problems = ["one model key: the perturbed file was compared with its own record"] if (
        against["same_model"] or against["baseline"] == result.model_key) else []
    heads = (against["suites"].get(suite) or {}).get("heads") or {}
    entry = (heads.get("comparison") or {}).get(POSITIVE_METRIC)
    change = ((heads.get("changes") or {}).get("metrics") or {}).get(POSITIVE_METRIC) or {}
    delta, verdict, n = (entry or {}).get("delta"), (entry or {}).get("verdict"), (entry or {}).get("n_paired") or 0
    fixed, broken = change.get("fixed") or 0, change.get("broken") or 0
    # the metric's rate on the paired reads: the baseline's is the room the perturbed model had to get worse
    rates = {"baseline": (entry or {}).get("mean_a_paired"), "this": (entry or {}).get("mean_b_paired")}
    if entry is None or not n:
        problems.append(f"{POSITIVE_METRIC} was not compared on any read")
    else:
        if delta is None or delta >= 0:
            problems.append(f"Δ {_signed(delta)} on {POSITIVE_METRIC}: the model with its {POSITIVE_MODULE} zeroed "
                            "must be worse (Δ = this − the baseline < 0)")
        if verdict != "a_better":
            problems.append(f"verdict {verdict} on {POSITIVE_METRIC}, not a_better (baseline better)")
        if fixed + broken == 0:
            problems.append(f"no read fixed or broken on {POSITIVE_METRIC}")
        elif broken <= fixed:
            problems.append(f"{fixed} reads fixed and {broken} broken on {POSITIVE_METRIC}: broken must outnumber "
                            "fixed")
    headline = ((((against.get("summary") or {}).get("suites") or {}).get(suite) or {}).get("headline") or {}).get(
        POSITIVE_METRIC) or {}
    if (headline.get("delta"), headline.get("verdict")) != (delta, verdict):
        problems.append(f"the summary's {POSITIVE_METRIC} (Δ {_signed(headline.get('delta'))}, verdict "
                        f"{headline.get('verdict')}) is not the comparison's")
    detail = layout.diagnostics_against_md(result.model_key, suite, against["baseline"])
    rows = [line for line in (detail.read_text(encoding="utf-8").splitlines() if detail.is_file() else [])
            if line.startswith(f"| {POSITIVE_METRIC} |")]
    rendered = AGAINST_VERDICTS["a_better"]
    if not rows or not rows[0].rstrip(" |").endswith(rendered):
        problems.append(f"the suite detail's {POSITIVE_METRIC} row does not read {rendered!r} ({detail.name})")
    evidence = {"summary": f"{POSITIVE_MODULE} zeroed: {POSITIVE_METRIC} {_percent(rates['baseline'])} → "
                           f"{_percent(rates['this'])}, Δ {_signed(delta)} on {n} reads, "
                           f"{AGAINST_VERDICTS.get(verdict, verdict)}, {fixed} fixed / {broken} broken ({suite}, "
                           f"perturbed {result.model_key} against {against['baseline']})",
                "suite": suite, "module": POSITIVE_MODULE, "zeroed": zeroed, "metric": POSITIVE_METRIC,
                "model_key": result.model_key, "baseline": against["baseline"], "delta": delta, "verdict": verdict,
                "ci95": (entry or {}).get("ci95"), "rates": rates, "fixed": fixed, "broken": broken,
                "detail": str(detail), "problems": problems}
    if problems:
        return check(name, "fail", n, evidence, f"the positive control did not see the perturbed head get worse "
                                                f"({'; '.join(problems[:3])}): the comparison is blind or backwards")
    return check(name, "pass", n, evidence)


def checkpoint_model_key(checkpoint: str) -> str:
    """The model key ``diagnose`` files the checkpoint ``checkpoint`` under in this process
    (``diagnostics.identity.model_key`` of its bytes' digest and the AlignAIR code)."""
    from alignair_assay.diagnostics.identity import alignair_code, model_key, weights_sha256

    return model_key(weights_sha256(checkpoint), alignair_code()["code_sha256"])


def all_vs_diagnose(cfg: AuditConfig, layout: RunLayout) -> dict:
    """``agreement.all_vs_diagnose``: on the shared suite's one build in ``layout`` (``shared_suite``), ``all``'s
    per-case end-to-end scores of ``AGREEMENT_TOOL`` (``layout.tool_scores``) against ``diagnose``'s per-case
    baseline scores of the same checkpoint (``workflow.diagnose.read_case_scores`` under ``checkpoint_model_key``),
    whose record must be of this build (its cases' and truth's digests). The same cases must be scored (an excluded
    case by neither) and every metric both surfaces score must have the same value on every case (NaN on both, or
    absent from both, is the same). ``n`` is the cases compared; the evidence names the first differing ones."""
    from alignair_assay.workflow.diagnose import read_case_scores

    name, suite = "agreement.all_vs_diagnose", shared_suite(cfg)
    if suite is None:
        return check(name, "fail", 0, {"summary": "no suite both null surfaces build"},
                     "the null all and diagnose suites share no suite to compare on")
    evidence = {"suite": suite, "tool": AGREEMENT_TOOL}
    problems = []
    path = layout.tool_scores(suite, "shared", AGREEMENT_TOOL)
    key = checkpoint_model_key(cfg.checkpoint)
    record = read_json(layout.diagnostics_suite_json(key, suite))
    evidence["model_key"] = key
    if not path.is_file():
        problems.append(f"all did not score {AGREEMENT_TOOL} on {suite} ({path.name} is missing)")
    if record is None:
        problems.append(f"no diagnose record of {suite} for the checkpoint ({key})")
    else:
        build = (record.get("key") or {}).get("suite") or {}
        files = {"cases_sha256": layout.cases(suite), "truth_sha256": layout.truth(suite)}
        if any(not file.is_file() or build.get(field) != sha256_file(file) for field, file in files.items()):
            problems.append(f"diagnosed on another build of {suite} than the one all scored")
    if problems:
        evidence.update(summary="; ".join(problems), problems=problems)
        return check(name, "fail", 0, evidence, "; ".join(problems))
    ends = read_scores(path)
    replayed = {case: scores["baseline"] for case, scores in read_case_scores(layout, key, suite).items()
                if scores["baseline"] is not None}
    only_all = [case for case in ends if case not in replayed]
    only_diagnose = [case for case in replayed if case not in ends]
    common = [case for case in ends if case in replayed]
    metrics = sorted({metric for scores in ends.values() for metric in scores}
                     & {metric for scores in replayed.values() for metric in scores})
    differ = [(case, metric, ends[case].get(metric), replayed[case].get(metric))
              for case in common for metric in metrics
              if not _same_score(ends[case].get(metric), replayed[case].get(metric))]
    cases_differing = list(dict.fromkeys(case for case, *_ in differ))
    problems += [f"cases only all scores: {len(only_all)} (e.g. {', '.join(only_all[:3])})"] if only_all else []
    problems += ([f"cases only diagnose scores: {len(only_diagnose)} (e.g. {', '.join(only_diagnose[:3])})"]
                 if only_diagnose else [])
    problems += [] if metrics else ["no metric is scored by both surfaces"]
    problems += [f"{case} {metric}: all {_score_text(a)}, diagnose {_score_text(b)}" for case, metric, a, b in
                 differ[:_EXAMPLES]]
    evidence.update(metrics=metrics, cases=len(common), only_all=len(only_all), only_diagnose=len(only_diagnose),
                    differing=len(cases_differing), differing_values=len(differ),
                    first_differing=cases_differing[:_EXAMPLES], problems=problems,
                    summary=(f"{suite}: {AGREEMENT_TOOL}'s scores in all and the checkpoint's ({key}) in diagnose, "
                             f"{len(common)} cases × {len(metrics)} metrics: "
                             + (f"{len(cases_differing)} cases differ (e.g. {', '.join(cases_differing[:3])})"
                                if cases_differing else "identical case by case")
                             + (f"; {len(only_all)} only in all, {len(only_diagnose)} only in diagnose"
                                if only_all or only_diagnose else "")))
    if problems:
        return check(name, "fail", len(common), evidence,
                     f"all and diagnose score the checkpoint differently ({'; '.join(problems[:3])}): the CLI's AIRR "
                     "output and the in-process replay are not one pipeline, or did not run on one build")
    return check(name, "pass", len(common), evidence)


def _same_score(a, b) -> bool:
    """Two per-case values of one metric are the same: equal, both NaN, or both absent (None)."""
    if a is None or b is None:
        return a is b
    return a == b or (math.isnan(a) and math.isnan(b))


def _score_text(value) -> str:
    return "absent" if value is None else f"{value:g}"


def real_null(cfg: AuditConfig, scratch: RunLayout) -> dict:
    """``null.real``: ``real`` with the checkpoint as ``alignair-api@a`` and ``@b`` on ``cfg.real_donor``, capped at
    ``cfg.real_cap`` (``workflow.real.run_real``) in ``scratch``. Every checkpoint comparison (per donor and pooled,
    each arm: overall, per stratum, the sensitivity analyses and the clusterings) must be null, with no read fixed
    or broken, equal one-sided exclusions and informational summaries, and the two runs' own summaries equal."""
    tools = tuple(f"alignair-api@{label}" for label in NULL_LABELS)
    real_cfg = RealConfig(donors=(cfg.real_donor,), split="dev", data_root=cfg.data_root, cap=cfg.real_cap,
                          tools=tools, threads=cfg.threads, device=cfg.device, reference=cfg.reference,
                          checkpoints={label: cfg.checkpoint for label in NULL_LABELS})
    summary = run_real(real_cfg, scratch)
    problems = [f"{stopped['donor']} stopped at {stopped['step']}: {stopped['reason']}"
                for stopped in summary.get("stopped_donors") or []]
    problems += [f"{failed['tool']} on {failed['donor']}'s {failed['arm']} arm failed: {failed['error']}"
                 for failed in summary.get("failed_runs") or []]
    block = (summary.get("donors") or {}).get(cfg.real_donor) or {}
    arms, reads = {}, 0
    comparisons = [(f"{cfg.real_donor}", entry) for entry in block.get("comparisons") or []]
    comparisons += [("pooled", entry) for entry in (summary.get("pooled") or {}).get("comparisons") or []]
    for where, entry in comparisons:
        if entry.get("kind") != CHECKPOINT_PAIR:
            continue
        label = f"{where} {entry['arm']}"
        if entry.get("status") != "compared":
            problems.append(f"{label}: not compared ({entry.get('reason')})")
            continue
        problems += real_comparison_problems(entry, label)
        n, short = pairing(entry.get("overall") or {}, REAL_HEADLINE, f"{label} overall")
        problems += short
        if where != "pooled":
            arms[entry["arm"]] = {"reads": entry.get("n_reads"), "n": n, "metrics": len(entry.get("overall") or {}),
                                  "paired_metrics": _paired(entry.get("overall") or {}),
                                  "fixed": ((entry.get("changes") or {}).get("reads") or {}).get("fixed"),
                                  "broken": ((entry.get("changes") or {}).get("reads") or {}).get("broken")}
            reads += n if entry["arm"] == "shared" else 0
    for arm, runs in (block.get("tools") or {}).items():
        first, second = (runs.get(tool) for tool in tools)
        if first is not None and first != second:
            problems.append(f"{cfg.real_donor} {arm}: the two runs' own R1 summaries differ")
    for arm in ("shared", "genotype"):
        if arm not in arms:
            problems.append(f"{cfg.real_donor} {arm}: no checkpoint comparison")
        else:
            pair = [scratch.real_predictions(tool, cfg.real_donor, arm) for tool in tools]
            arms[arm]["predictions_identical"] = (all(path.is_file() for path in pair)
                                                  and sha256_file(pair[0]) == sha256_file(pair[1]))
    oracle = (block.get("oracle_self_check") or {}).get("status")
    evidence = {"summary": f"{cfg.real_donor} capped at {cfg.real_cap}: " + ", ".join(
                    f"{arm} {entry['metrics']} metrics over {entry['reads']} reads (headline metrics pair at least "
                    f"{entry['n']}), fixed {entry['fixed']} / broken {entry['broken']}" for arm, entry in arms.items())
                + f"; its self-check {oracle}",
                "donor": cfg.real_donor, "cap": cfg.real_cap, "tools": list(tools), "arms": arms,
                "oracle_self_check": oracle}
    return _null_check("null.real", problems, reads, evidence, "real's comparison of one checkpoint with itself")


def real_comparison_problems(entry: dict, where: str) -> list[str]:
    """``compare_problems`` over every comparison of one ``real`` pair (overall, per stratum, the sensitivity
    analyses, the clusterings), ``changes_problems`` of its reads fixed and broken, and the two runs' one-sided
    exclusions and informational summaries, which must be equal."""
    problems = compare_problems(entry.get("overall") or {}, f"{where} overall")
    for label, found in (entry.get("strata") or {}).items():
        problems += compare_problems(found, f"{where} {label}")
    for label, found in (entry.get("sensitivity") or {}).items():
        problems += compare_problems(found.get("comparison") or {}, f"{where} sensitivity {label}")
    for label, found in (entry.get("clusterings") or {}).items():
        problems += compare_problems(found.get("comparison") or {}, f"{where} clustering {label}")
    problems += changes_problems(entry.get("changes"), where)
    one_sided = entry.get("one_sided") or {}
    if one_sided.get("a") != one_sided.get("b"):
        problems.append(f"{where}: the two runs exclude other reads on their own")
    for metric, found in (entry.get("informational") or {}).items():
        if found.get("a") != found.get("b"):
            problems.append(f"{where} {metric}: the informational summaries differ")
    return problems
