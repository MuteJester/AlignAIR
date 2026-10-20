"""The run summary: plain, JSON-ready data describing what a run directory's files say, for the run report.

Everything is read back from the run directory (build records, run records, score files and comparisons), so
``all`` and ``report`` describe the same state. ``report.run.run_report_markdown`` renders the summary.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import sha256_file
from alignair_assay.core.reference import Reference
from alignair_assay.metrics import is_informational, lower_is_better
from alignair_assay.report.run import RUN_REPORT_KIND, RUN_REPORT_VERSION
from alignair_assay.runners.base import read_run_json, require_successful_run
from alignair_assay.scoring.oracle import oracle_problems
from alignair_assay.scoring.pipeline import read_scores
from alignair_assay.stats import compare
from alignair_assay.suites import get_suite
from alignair_assay.truth.io import read_truth_jsonl
from alignair_assay.workflow.build import build_is_current

if TYPE_CHECKING:
    from alignair_assay.workflow.orchestrate import RunConfig

ORACLE = "oracle"
# Family-wise error rate of every Holm-corrected verdict in a run.
ALPHA = 0.05
VERDICTS = ("a_better", "b_better", "tie_or_inconclusive")
# S1's stratum with S10's read design (full) and SHM rate (igg) under the default distributions.
S10_BASELINE = "full_igg"
_EXAMPLES = 5


def expected_runs(cfg: RunConfig) -> list[tuple[str, str, str]]:
    """The ``(tool, suite, arm)`` runs ``all`` makes for ``cfg``, in order: the oracle on every suite's shared arm,
    then each of the two tools on every suite and each of the suite's arms (the oracle's shared runs once)."""
    jobs = [(ORACLE, suite, "shared") for suite in cfg.suites]
    jobs += [(tool, suite, arm) for tool in cfg.tools for suite in cfg.suites for arm in get_suite(suite).arms]
    return list(dict.fromkeys(jobs))


def inputs_unchanged(layout: RunLayout, suite: str, provenance: dict) -> bool:
    """True when a run's recorded input files still have their recorded SHA-256: the suite's reads (which every
    run record holds) and, when recorded, its truth (the oracle) and donors (the genotype arm)."""
    if "reads" not in provenance:
        return False
    files = {"reads": layout.reads(suite), "truth": layout.truth(suite), "donors_json": layout.donors(suite)}
    for key, path in files.items():
        if key in provenance:
            recorded = (provenance[key] or {}).get("sha256")
            if not path.is_file() or sha256_file(path) != recorded:
                return False
    return True


def run_status(layout: RunLayout, tool: str, suite: str, arm: str, settings: dict | None) -> dict:
    """Whether a run may be reused and scored. ``status`` is ``ok`` when ``runners.base.require_successful_run``
    passes, the predictions exist, the record's ``settings`` (the tool's fingerprint when it ran) equal
    ``settings`` (the run directory's configuration for the tool) and its inputs are unchanged
    (``inputs_unchanged``). Otherwise it is ``missing`` (no run record or no predictions), ``failed`` (a non-zero
    return code, or an unreadable record) or ``stale`` (made with other tool settings, or on files that changed
    since). The dict also holds ``tool``, ``suite``, ``arm``, the ``returncode``, the ``failed_donor`` and
    ``error`` the record names (None otherwise) and the ``run_json`` path relative to the run directory."""
    path = layout.run_json(tool, suite, arm)
    status = {"tool": tool, "suite": suite, "arm": arm, "status": "ok", "returncode": None, "failed_donor": None,
              "error": None, "run_json": relative(layout, path)}
    try:
        record = require_successful_run(path)
    except FileNotFoundError:
        return {**status, "status": "missing", "error": "no run record: the run never finished"}
    except (ValueError, TypeError) as error:
        try:
            record = read_run_json(path)
        except (ValueError, TypeError):
            return {**status, "status": "failed", "error": f"unreadable run record: {error}"}
        return {**status, "status": "failed", "returncode": record.returncode,
                "failed_donor": record.provenance.get("failed_donor"), "error": record.provenance.get("error")}
    status["returncode"] = record.returncode
    if not layout.predictions(tool, suite, arm).is_file():
        return {**status, "status": "missing", "error": "the run's predictions are missing"}
    if record.provenance.get("settings") != settings:
        return {**status, "status": "stale",
                "error": "made with other tool settings than the run directory's configuration; run all again"}
    if not inputs_unchanged(layout, suite, record.provenance):
        return {**status, "status": "stale", "error": "the suite's files changed after this run; run all again"}
    return status


def configured_settings(layout: RunLayout) -> dict[str, dict]:
    """Per tool, the settings (fingerprint) the run directory's ``manifest.json`` records; empty without one."""
    return ((read_json(layout.manifest) or {}).get("identity") or {}).get("settings") or {}


def run_statuses(layout: RunLayout, cfg: RunConfig) -> dict[tuple[str, str, str], dict]:
    """``run_status`` of every expected run (``expected_runs``) against the manifest's tool settings
    (``configured_settings``). A run that would be ``ok`` on a suite whose build does not match ``cfg``
    (``workflow.build.build_is_current``) is ``stale`` too."""
    settings = configured_settings(layout)
    digest = Reference.from_dataconfig(cfg.reference).digest()
    current = {suite: build_is_current(layout, suite, split=cfg.split, reference_digest=digest, seed=cfg.seed,
                                       scale=cfg.scale) for suite in cfg.suites}
    statuses = {}
    for tool, suite, arm in expected_runs(cfg):
        status = run_status(layout, tool, suite, arm, settings.get(tool))
        if status["status"] == "ok" and not current[suite]:
            status = {**status, "status": "stale",
                      "error": "the suite's build does not match the run directory's configuration; run all again"}
        statuses[tool, suite, arm] = status
    return statuses


def collect_run_summary(layout: RunLayout, cfg: RunConfig, *, scoring_errors: list[dict] | None = None) -> dict:
    """The run report's data for ``cfg`` over ``layout``, as strict JSON.

    It holds ``kind``, ``version``, the compared tools ``a`` and ``b``, the ``config`` (suites, tools, split,
    reference, seed, scale) and the manifest's ``config_hash``. ``failed_runs`` lists every expected run whose
    status (``run_statuses``) is not ``ok``, then ``scoring_errors`` (runs that succeeded but could not be
    scored). ``suites`` has, per suite: ``name``, ``title``, ``purpose``, ``primary``, ``cases``, ``excluded``
    (from ``build.json``), the ``oracle`` check (``status`` perfect, imperfect or not scored; ``cases``,
    ``problems`` and the first ``examples`` as ``[case id, metric, value]``) and per arm whether the two tools were
    ``compared``, the ``link`` to the arm's ``comparison.md`` (relative to the run directory), the primary-metric
    ``rows`` of ``comparison.json`` and their ``verdicts`` counts. ``oracle_failures`` repeats the suites whose
    oracle is not perfect. ``s10_vs_s1``, ``s11_parity`` and ``speed`` are None without their data, and
    ``genotype_arms`` is empty without a genotype arm (see the functions below).
    """
    a, b = cfg.tools
    failed = [status for status in run_statuses(layout, cfg).values() if status["status"] != "ok"]
    suites = [_suite_summary(layout, cfg, name) for name in cfg.suites]
    manifest = read_json(layout.manifest) or {}
    return {
        "kind": RUN_REPORT_KIND, "version": RUN_REPORT_VERSION, "a": a, "b": b,
        "config": {"suites": list(cfg.suites), "tools": list(cfg.tools), "split": cfg.split,
                   "reference": cfg.reference, "seed": cfg.seed, "scale": cfg.scale},
        "config_hash": manifest.get("config_hash"),
        "failed_runs": failed + list(scoring_errors or []),
        "oracle_failures": [{"suite": suite["name"], **suite["oracle"]} for suite in suites
                            if suite["oracle"]["status"] != "perfect"],
        "suites": suites,
        "s10_vs_s1": s10_vs_s1(layout, cfg),
        "s11_parity": s11_parity(layout, cfg),
        "genotype_arms": genotype_arms(layout, cfg),
        "speed": read_json(layout.speed_json),
    }


def oracle_check(layout: RunLayout, suite: str) -> dict:
    """The oracle rule (``scoring.oracle.oracle_problems``) on the oracle's scores of ``suite``'s shared arm:
    ``status`` (``perfect``, ``imperfect``, or ``not scored`` when there are no oracle scores), the scored
    ``cases``, the number of ``problems`` and the first ``examples`` (``[case id, metric, value]``, nan as None).
    A suite with no scored case is imperfect."""
    path = layout.tool_scores(suite, "shared", ORACLE)
    if not path.is_file():
        return {"status": "not scored", "cases": 0, "problems": 0, "examples": []}
    scores = read_scores(path)
    if not scores:
        return {"status": "imperfect", "cases": 0, "problems": 1, "examples": [[None, "no scored case", None]]}
    problems = oracle_problems(scores)
    examples = [[case_id, metric, None if math.isnan(value) else value]
                for case_id, metric, value in problems[:_EXAMPLES]]
    return {"status": "imperfect" if problems else "perfect", "cases": len(scores), "problems": len(problems),
            "examples": examples}


def s10_vs_s1(layout: RunLayout, cfg: RunConfig) -> dict | None:
    """S10's primary-metric means against S1's per tool, when both suites are in the run: unpaired and
    informational (no test). The baseline is S1's ``S10_BASELINE`` stratum, whose read design and SHM rate are
    S10's. Per tool, rows hold the ``metric``, the summary entries (``mean``, ``n``, ``n_applicable``,
    ``coverage``; None when absent) ``s1`` and ``s10``, the ``delta`` of their means (S10 - S1) and the entries per
    held-out setting (``by_setting``). None when neither tool has both score reports."""
    if not {"S1", "S10"} <= set(cfg.suites):
        return None
    s10 = get_suite("S10")
    settings = [stratum.labels.get("heldout", stratum.name) for stratum in s10.strata]
    tools = {}
    for tool in cfg.tools:
        s1_report = read_json(layout.tool_report_json("S1", "shared", tool))
        s10_report = read_json(layout.tool_report_json("S10", "shared", tool))
        if s1_report is None or s10_report is None:
            continue
        baseline = s1_report["by_slice"].get("stratum", {}).get(S10_BASELINE)
        heldout = s10_report["by_slice"].get("heldout", {})
        rows = []
        for metric in s10.primary:
            before, after = _entry(baseline, metric), _entry(s10_report["overall"], metric)
            means = [None if entry is None else entry.get("mean") for entry in (before, after)]
            rows.append({"metric": metric, "s1": before, "s10": after,
                         "delta": None if None in means else means[1] - means[0],
                         "by_setting": {setting: _entry(heldout.get(setting), metric) for setting in settings}})
        tools[tool] = rows
    return {"baseline": f"S1 {S10_BASELINE}", "settings": settings, "tools": tools} if tools else None


def s11_parity(layout: RunLayout, cfg: RunConfig) -> dict | None:
    """Per tool, its S11 forward reads against their reverse-complemented twins (cases sharing ``groups["pair"]``),
    compared with ``stats.compare`` (forward as a, reverse as b) over every non-informational metric, Holm at
    ``ALPHA``. Per tool: the ``pairs`` with both cases scored, the ``family_size``, the metrics whose verdict
    ``differ`` and the S11 primary-metric ``rows`` (paired ``forward`` and ``reverse`` means, ``delta`` = reverse -
    forward, ``n_paired``, ``verdict``). None without S11 or without scores."""
    if "S11" not in cfg.suites or not layout.truth("S11").is_file():
        return None
    pairs: dict[str, dict[int, str]] = {}
    for truth in read_truth_jsonl(layout.truth("S11")):
        if "pair" in truth.groups:
            pairs.setdefault(truth.groups["pair"], {})[truth.orientation] = truth.case_id
    primary = get_suite("S11").primary
    tools = {}
    for tool in cfg.tools:
        path = layout.tool_scores("S11", "shared", tool)
        if not path.is_file():
            continue
        scores = read_scores(path)
        forward, reverse = {}, {}
        for pair, cases in pairs.items():
            if cases.get(0) in scores and cases.get(1) in scores:
                forward[pair], reverse[pair] = scores[cases[0]], scores[cases[1]]
        seen = {name for side in (forward, reverse) for case in side.values() for name in case}
        family = sorted(name for name in seen if not is_informational(name))
        comparison = compare(forward, reverse, family, alpha=ALPHA,
                             lower_is_better={name for name in family if lower_is_better(name)})
        rows = [{"metric": metric, "forward": comparison[metric]["mean_a_paired"],
                 "reverse": comparison[metric]["mean_b_paired"], "delta": comparison[metric]["delta"],
                 "n_paired": comparison[metric]["n_paired"], "verdict": comparison[metric]["verdict"]}
                for metric in primary if metric in comparison]
        tools[tool] = {"pairs": len(forward), "family_size": len(family),
                       "differ": [name for name in family if comparison[name]["verdict"] != "tie_or_inconclusive"],
                       "rows": rows}
    return {"alpha": ALPHA, "tools": tools} if tools else None


def genotype_arms(layout: RunLayout, cfg: RunConfig) -> list[dict]:
    """Per suite with a genotype arm: its ``arms`` and, per primary metric, each tool's summary entry in each arm
    (``values[tool][arm]``: ``mean``, ``n``, ``n_applicable``, ``coverage``; None when the tool has no score report
    for that arm or no value of the metric)."""
    out = []
    for name in cfg.suites:
        suite = get_suite(name)
        if "genotype" not in suite.arms:
            continue
        overall = {(tool, arm): (read_json(layout.tool_report_json(name, arm, tool)) or {}).get("overall")
                   for tool in cfg.tools for arm in suite.arms}
        rows = [{"metric": metric, "values": {tool: {arm: _entry(overall[tool, arm], metric) for arm in suite.arms}
                                              for tool in cfg.tools}} for metric in suite.primary]
        out.append({"suite": name, "arms": list(suite.arms), "rows": rows})
    return out


def read_json(path: Path) -> dict | None:
    """The JSON object in ``path``; None when the file is missing or not valid JSON."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def relative(layout: RunLayout, path: Path) -> str:
    """``path`` relative to the run directory, with forward slashes (as a markdown link)."""
    return Path(path).relative_to(layout.root).as_posix()


def _suite_summary(layout: RunLayout, cfg: RunConfig, name: str) -> dict:
    suite = get_suite(name)
    build = read_json(layout.build_json(name)) or {}
    excluded = (build.get("truth") or {}).get("excluded") or {}
    return {"name": name, "title": suite.title, "purpose": suite.purpose, "primary": list(suite.primary),
            "cases": build.get("n_cases"), "excluded": sum(excluded.values()),
            "oracle": oracle_check(layout, name),
            "arms": {arm: _arm_summary(layout, cfg, name, arm, suite.primary) for arm in suite.arms}}


def _arm_summary(layout: RunLayout, cfg: RunConfig, name: str, arm: str, primary) -> dict:
    """The primary-metric rows of the arm's comparison of the run's two tools; not ``compared`` when there is no
    comparison of exactly those tools."""
    report = read_json(layout.comparison_json(name, arm))
    if report is None or (report.get("a"), report.get("b")) != tuple(cfg.tools):
        return {"compared": False, "link": None, "rows": {}, "verdicts": None}
    rows = {metric: report["comparison"][metric] for metric in primary if metric in report["comparison"]}
    verdicts = {verdict: sum(row["verdict"] == verdict for row in rows.values()) for verdict in VERDICTS}
    return {"compared": True, "link": relative(layout, layout.comparison_md(name, arm)), "rows": rows,
            "verdicts": verdicts}


def _entry(entries: dict | None, metric: str) -> dict | None:
    """The summary entry (``mean``, ``n``, ``n_applicable``, ``coverage``) of ``metric`` in a ``{metric:
    summary}`` mapping of a report; None when absent."""
    return (entries or {}).get(metric)
