"""The diagnostics report of one model: the plain dict ``workflow.diagnose`` builds, its one-page summary and its
markdown.

``build_diagnostics_report`` assembles the dict (strict JSON: NaN and infinity become None) with its ``summary``
(``diagnostics_summary``), which ``diagnostics_scalars`` flattens for a training dashboard. With a comparison against
a baseline model (``against``: ``workflow.diagnose.compare_models``; Δ = this model − the baseline, fixed: reads the
baseline gets wrong and this model right), the report holds it and its summary (``against_summary``), both
rendered by ``report.diagnostics_against``. Three renderings here:

- ``diagnostics_markdown``: the summary, ``diagnostics.md``. A three-line header, then, with a comparison,
  ``AGAINST_SECTION`` first (``report.diagnostics_against.against_section``). Then ``SUMMARY_SECTIONS``:
  1. **Where the end-to-end error comes from**: per suite, for ``clone_key.exact`` and ``junction.nt_exact``, the
     baseline, the Δ of each stage's truth (bold when significant, marked when worse), what is left after every
     stage's truth, the soft decoder's Δ and the paired count (with its clusters);
  2. **Calls**: per suite and gene, the baseline top-1 in set, the true calls' Δ, the head's top-1 → top-3 and ECE;
  3. **Boundaries: decoder vs network**: each edge's exact rate under the soft-argmax → the logits' argmax, which the product decodes with;
  4. **Metadata heads**: orientation, productive, mutation rate and indel count;
  5. **Speed**: reads per second and the top stages and modules;
  6. **Flags**: at most ``MAX_FLAGS`` bullets on what qualifies the numbers above.
- ``suite_markdown``: one suite's detail, ``<S>/report.md``: its reads, then ``SUITE_SECTIONS`` (head metrics with
  soft and argmax side by side, the stage contributions with the paired means beside each Δ, calibration, the single
  swaps, fallbacks and crops), each always present (a section without data says so).
- ``profile_markdown``: the speed profile's tables, ``profile.md``.

Each suite's detail of a comparison, ``<S>/against_<other key>.md``, is
``report.diagnostics_against.against_markdown``.
"""
from __future__ import annotations

from alignair_assay.report.build import clean
from alignair_assay.report.diagnostics_against import AGAINST_SECTION, against_section, against_summary
from alignair_assay.report.markdown import (
    CLUSTER_NOTE,
    MISSING,
    format_cell,
    format_ci,
    format_clusters,
    format_paired,
    format_value,
    is_clustered,
    labelled,
    markdown_table,
)

DIAGNOSTICS_KIND = "alignair_assay.diagnostics_report"
DIAGNOSTICS_VERSION = "3"
SUMMARY_SECTIONS = ("Where the end-to-end error comes from", "Calls", "Boundaries: decoder vs network",
                    "Metadata heads", "Speed", "Flags")
SUITE_SECTIONS = ("Head metrics", "Stage contributions", "Calibration", "Single-swap variants", "Fallbacks and crops")
# The end-to-end metrics the summary traces through the stages.
SUMMARY_METRICS = ("clone_key.exact", "junction.nt_exact")
# The cumulative chain's stages, in order, and their summary column names.
STAGES = ("orientation", "segmentation", "allele_calls", "germline_alignment")
STAGE_LABELS = {"orientation": "orientation", "segmentation": "segmentation", "allele_calls": "calls",
                "germline_alignment": "germline"}
# The metadata heads the summary shows, by summary column.
METADATA_HEADS = {"Orientation": "head.orientation.correct", "Productive": "head.productive.correct",
                  "Mutation rate abs. err": "head.mutation_rate_abs_err.read",
                  "Indel count abs. err": "head.indel_count_abs_err"}
# A D presence head's columns of the summary's metadata table, shown when a suite's model has the head: its accuracy
# against the truth's D class, and the share of truth-callable Ds it gated away (their D call dropped).
D_PRESENCE_HEADS = {"D presence": "head.d_presence.correct", "Callable D gated": "head.d_presence.callable_gated"}
# Flags: a baseline coverage below this, a fallback reason above this share of a suite's reads; at most this many.
COVERAGE_FLAG = 0.95
FALLBACK_FLAG = 0.05
MAX_FLAGS = 8
# Items named per flag before the rest are counted.
FLAG_ITEMS = 6
# A comparison's verdict, from the "to" variant's side (a step adds truth; a swap changes the baseline).
VERDICTS = {"to_better": "better", "from_better": "worse", "tie_or_inconclusive": "tie"}
HOLM_NOTE = ("Holm is per comparison (one step or swap of one suite, over its metrics), not across steps × suites: "
             "among many comparisons, expect a few false verdicts.")
_GENES = ("v", "d", "j")
_EDGES = ("start", "end")
_BOUNDARY_COLUMNS = (("exact", "soft"), ("exact", "argmax"), ("abs_err", "soft"), ("abs_err", "argmax"),
                     ("err", "soft"), ("err", "argmax"))
_ALLELE_METRICS = ("allele_top1", "allele_top3", "allele_top5", "allele_top1_generative", "n_above_threshold")
_META_METRICS = ("head.mutation_rate_abs_err.read", "head.mutation_rate_abs_err.record", "head.indel_count_abs_err",
                 "head.productive.correct", "head.orientation.correct", "head.d.short_d_prob_ok",
                 "head.d_presence.correct", "head.d_presence.callable_gated", "head.cropped")
_READS = ("built", "diagnosed", "scored", "excluded", "cropped")

_ERRORS_NOTE = ("Each Δ is the change when the truth of one more stage is fed in (in percentage points), given "
                "perfect upstream stages; the steps need not add up. Bold: significant (Holm within the step); "
                "(worse): significantly worse. true_germline: every stage's truth fed in, with the gap left to 100% "
                "(the metadata heads and the AIRR assembly). soft_decoder: the product's argmax boundaries "
                "replaced by the soft-argmax expectations, nothing else changed.")
_CALLS_NOTE = ("Each cell is V / D / J. Top-1 in set: the product's call at baseline. Δ true calls: feeding the "
               "truth call after the truth orientation and spans (bold: significant). Head: the calibrated allele "
               "head's top-1 → top-3 in the truth set. ECE: the top-1 probability's expected calibration error.")
_BOUNDARIES_NOTE = ("Exact-boundary rate (%) under the soft-argmax decoder → the position logits' argmax, the product's decoder. "
                    "A large jump is the decoder, not the network.")
_HEADS_NOTE = ("The raw head outputs of the product pipeline (the baseline replay), scored against observable "
               "truth before any post-processing. Boundaries are 0-based and end-exclusive. soft: the floor of the "
               "soft-argmax expectation, the decoder before the argmax; argmax: the position logits' argmax, the "
               "product's decoder, whose end can reach the read end (the soft-argmax cannot). Exact: the edge is inside the truth interval; "
               "errors are in nucleotides, a signed one negative before the interval. Columns marked "
               "informational are not tested.")
_CONTRIBUTIONS_NOTE = ("The truth of one more stage is fed in at each step (baseline → true_orientation → "
                       "true_spans → true_calls → true_germline); each variant is scored like a tool and compared "
                       "with the previous one on the cases both cover, Holm-corrected over the metrics. A step's Δ "
                       "is the error that stage contributes given perfect upstream stages, not a share of the "
                       "baseline's error, so the steps need not add up. Verdicts: better (the truth improves the "
                       "metric significantly), worse, tie (no significant difference). Each step's table gives each "
                       "variant's own mean over its own cases (n, coverage) and the paired means, whose difference "
                       "is Δ; rates are in percent, their differences in percentage points.")
_CALIBRATION_NOTE = ("Reliability of each gene's top-1 calibrated allele probability against the top-1 being in the "
                     "scored truth set, in equal-width bins. ECE: the bins' read-weighted |accuracy − confidence|. "
                     "A bin's cell is reads: accuracy / mean confidence; empty bins are left out.")
_SWAPS_NOTE = ("Each swap changes one thing on the baseline (soft_decoder: the segments from the soft-argmax "
               "expectations; true_classifier_spans: the truth spans into the classifier only, the segments "
               "unchanged) and is compared with the baseline like a step above. better: the swap improves the "
               "metric significantly.")
_PROFILE_NOTE = ("Stages: the replay's stages, with their share of the whole run. Modules: the network's leaf "
                 "modules, with their share of the forward stage, from separate runs with timing hooks; other is "
                 "the forward time spent outside them.")
_FALLBACKS_NOTE = ("Cropped: reads longer than AlignAIR's window, which it crops; their edges past the cut are not "
                   "scored. A fallback is a read whose truth for a segment could not be substituted in a variant "
                   "(unscorable, unobserved, a wrong model orientation, cropped, an allele the model cannot call, an "
                   "inverted D): that segment kept the model's output.")
def build_diagnostics_report(*, model: dict, config: dict, suites: dict, profile: dict | None,
                             against: dict | None = None) -> dict:
    """The diagnostics report: ``kind``, ``version``, the ``model`` (``model_key``, ``source``,
    ``weights_sha256``, ``code_sha256``, ``alignair_source``, ``torch_version``, ``assay_code_sha256``), the run
    ``config`` (``run_dir``, ``suites``, ``device``, ``batch_size``, ``variants``, ``max_cases``, ``profile``, and
    when known ``seconds`` (the call's wall time), ``stale_builds`` (suite -> what changed), ``replay_verified`` and
    ``replay_changes``), the per-suite results by suite name, the ``profile`` (None when not measured), the
    comparison with another model (``against``: ``workflow.diagnose.compare_models``; None without one) and the
    ``summary`` (``diagnostics_summary``), all strictly JSON-serializable. The inputs are not changed."""
    report = {"kind": DIAGNOSTICS_KIND, "version": DIAGNOSTICS_VERSION, "model": clean(model),
              "config": clean(config), "suites": clean(suites), "profile": clean(profile), "against": clean(against)}
    report["summary"] = clean(diagnostics_summary(report))
    return report


# --- the summary ------------------------------------------------------------------------------------------------

def diagnostics_summary(report: dict) -> dict:
    """The one-page summary of a report (built without ``summary``), as a plain dict:

    ``model_key``, ``source``, ``device``, ``batch_size``, ``split``, ``reads`` (suite -> diagnosed reads),
    ``computed`` and ``reused`` (suite names), ``seconds``, ``replay_verified`` and ``replay_changes`` (names);
    ``clustered`` (suite -> ``keys`` and ``clusters``, for the suites whose comparisons are cluster-aware: those
    with a cluster of two or more reads, as the suite's ``clusters`` record counts them);
    ``errors`` (metric of ``SUMMARY_METRICS`` -> suite -> ``baseline`` summary entry, ``steps`` (stage of
    ``STAGES`` -> the step's ``delta``, ``verdict``, ``test``, ``n_paired``, ``n_clusters`` and the ``stages`` it
    spans), ``true_germline`` (mean), ``gap`` (1 - that mean), ``soft_decoder`` (the swap's comparison entry, the
    same fields), ``n_paired`` and ``n_clusters`` (``[min, max]`` over those comparisons)); ``calls`` (suite -> gene
    -> ``top1_in_set``, ``true_calls`` (the step's entry), ``head.allele_top1``, ``head.allele_top3``, ``ece``);
    ``boundaries`` (suite -> ``head.{g}.{edge}_exact`` -> ``soft`` and ``argmax`` rates); ``metadata`` (suite ->
    head metric -> mean); ``speed`` (``reads_per_second``, ``device``, ``batch_size``, ``reads``, ``reused`` and the
    top three ``stages`` and ``modules``, ``[name, share]``; None without a profile); ``flags`` (at most
    ``MAX_FLAGS`` sentences); and ``against`` (``against_summary`` of the report's comparison; None without one). A
    missing value is None."""
    model, config, suites = report["model"], report["config"], report["suites"]
    summary = {
        "model_key": model["model_key"], "source": model.get("source"), "label": model.get("label"),
        "device": config.get("device"), "batch_size": config.get("batch_size"), "split": "dev",
        "reads": {name: suite["reads"]["diagnosed"] for name, suite in suites.items()},
        "computed": [name for name, suite in suites.items() if not suite.get("reused")],
        "reused": [name for name, suite in suites.items() if suite.get("reused")],
        "seconds": config.get("seconds"),
        "replay_verified": config.get("replay_verified", True) is not False,
        "replay_changes": list(config.get("replay_changes") or {}),
        "clustered": {name: {"keys": suite["clusters"]["keys"], "clusters": suite["clusters"]["clusters"]}
                      for name, suite in suites.items() if (suite.get("clusters") or {}).get("clusters")},
        "errors": {metric: {name: row for name, suite in suites.items() if (row := _error_row(suite, metric))}
                   for metric in SUMMARY_METRICS},
        "calls": {name: calls for name, suite in suites.items() if (calls := _calls(suite))},
        "boundaries": {name: _boundaries(suite) for name, suite in suites.items()},
        "metadata": {name: {head: _mean((suite.get("heads") or {}).get(head))
                            for head in (*METADATA_HEADS.values(), *D_PRESENCE_HEADS.values())}
                     for name, suite in suites.items()},
        "speed": _speed(report.get("profile")),
    }
    summary["flags"] = _flags(report, summary)
    summary["against"] = against_summary(report.get("against"))
    return summary


def diagnostics_scalars(summary: dict) -> dict[str, float]:
    """The summary's numbers flattened for a training dashboard (TensorBoard, W&B), None values left out:
    ``<S>/baseline/<metric>``, ``<S>/delta/<stage>/<metric>`` (the stages of ``STAGES`` and ``soft_decoder``),
    ``<S>/true_germline/<metric>``, ``<S>/heads/<head metric>`` (``head.{g}.{edge}_exact.{soft,argmax}``, the
    allele top-1/top-3 and the metadata heads), ``<S>/calibration/<g>.ece``, ``speed/reads_per_second`` and
    ``speed/{stages,modules}/<name>`` (shares); with a comparison, ``<S>/against/<metric>.delta`` for every compared
    metric (this model − the baseline) and ``<S>/against/<metric>.fixed`` (the baseline wrong, this model right) and
    ``.broken`` (reads) for every 0/1 one."""
    out: dict[str, float] = {}

    def put(key: str, value) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = float(value)

    for metric, rows in summary["errors"].items():
        for suite, row in rows.items():
            put(f"{suite}/baseline/{metric}", _mean(row["baseline"]))
            for stage, step in row["steps"].items():
                put(f"{suite}/delta/{stage}/{metric}", step["delta"])
            put(f"{suite}/true_germline/{metric}", row["true_germline"])
            put(f"{suite}/delta/soft_decoder/{metric}", (row["soft_decoder"] or {}).get("delta"))
    for suite, genes in summary["calls"].items():
        for g, calls in genes.items():
            put(f"{suite}/baseline/{g}.top1_in_set", calls["top1_in_set"])
            put(f"{suite}/delta/allele_calls/{g}.top1_in_set", (calls["true_calls"] or {}).get("delta"))
            for k in ("top1", "top3"):
                put(f"{suite}/heads/head.{g}.allele_{k}", calls[f"head.allele_{k}"])
            put(f"{suite}/calibration/{g}.ece", calls["ece"])
    for suite, edges in summary["boundaries"].items():
        for name, rates in edges.items():
            for decoder, value in rates.items():
                put(f"{suite}/heads/{name}.{decoder}", value)
    for suite, heads in summary["metadata"].items():
        for head, value in heads.items():
            put(f"{suite}/heads/{head}", value)
    speed = summary.get("speed")
    if speed:
        put("speed/reads_per_second", speed["reads_per_second"])
        for kind in ("stages", "modules"):
            for name, share in speed[kind]:
                put(f"speed/{kind}/{name}", share)
    for suite, compared in ((summary.get("against") or {}).get("suites") or {}).items():
        for metric, delta in compared["deltas"].items():
            put(f"{suite}/against/{metric}.delta", delta)
        for kind in ("fixed", "broken"):
            for metric, reads in compared[kind].items():
                put(f"{suite}/against/{metric}.{kind}", reads)
    return out


def _mean(entry) -> float | None:
    return entry.get("mean") if isinstance(entry, dict) else None


def _scores(suite: dict, variant: str) -> dict:
    return ((suite.get("variants") or {}).get(variant) or {}).get("scores") or {}


def _entry(entry: dict, stages: list[str] | None = None) -> dict:
    """The fields of a comparison entry the summary keeps."""
    kept = {key: entry.get(key) for key in ("delta", "verdict", "test", "n_paired", "n_clusters")}
    return {**kept, "stages": stages} if stages is not None else kept


def _error_row(suite: dict, metric: str) -> dict | None:
    baseline = _scores(suite, "baseline").get(metric)
    steps = {step["stages"][-1]: _entry(step["metrics"][metric], step["stages"])
             for step in suite.get("contributions") or [] if metric in step["metrics"]}
    swap = next((entry for comparison in suite.get("swaps") or [] if comparison["to"] == "soft_decoder"
                 for name, entry in comparison["metrics"].items() if name == metric), None)
    true_germline = _mean(_scores(suite, "true_germline").get(metric))
    if baseline is None and not steps and swap is None:
        return None
    compared = [*steps.values(), *([swap] if swap else [])]
    n_paired = [entry["n_paired"] for entry in compared if entry.get("n_paired") is not None]
    n_clusters = [entry["n_clusters"] for entry in compared if entry.get("n_clusters") is not None]
    return {"baseline": baseline, "steps": steps, "true_germline": true_germline,
            "gap": None if true_germline is None else 1.0 - true_germline,
            "soft_decoder": None if swap is None else _entry(swap),
            "n_paired": [min(n_paired), max(n_paired)] if n_paired else None,
            "n_clusters": [min(n_clusters), max(n_clusters)] if n_clusters else None}


def _calls(suite: dict) -> dict:
    heads, calibration = suite.get("heads") or {}, suite.get("calibration") or {}
    out = {}
    for g in _GENES:
        metric = f"{g}.top1_in_set"
        step = next((comparison["metrics"][metric] for comparison in suite.get("contributions") or []
                     if "allele_calls" in comparison["stages"] and metric in comparison["metrics"]), None)
        row = {"top1_in_set": _mean(_scores(suite, "baseline").get(metric)),
               "true_calls": None if step is None else _entry(step),
               "head.allele_top1": _mean(heads.get(f"head.{g}.allele_top1")),
               "head.allele_top3": _mean(heads.get(f"head.{g}.allele_top3")),
               "ece": (calibration.get(g) or {}).get("ece")}
        if any(value is not None for value in row.values()):
            out[g] = row
    return out


def _boundaries(suite: dict) -> dict:
    heads = suite.get("heads") or {}
    return {f"head.{g}.{edge}_exact": {decoder: _mean(heads.get(f"head.{g}.{edge}_exact.{decoder}"))
                                        for decoder in ("soft", "argmax")}
            for g in _GENES for edge in _EDGES}


def _speed(profile: dict | None) -> dict | None:
    if not profile:
        return None

    def top(breakdown: dict) -> list[list]:
        ranked = sorted(breakdown.items(), key=lambda item: (-item[1]["share"], item[0]))
        return [[name, entry["share"]] for name, entry in ranked[:3]]

    return {"reads_per_second": profile["reads_per_second"], "device": profile["device"],
            "batch_size": profile["batch_size"], "reads": profile["reads"], "reused": bool(profile.get("reused")),
            "stages": top(profile["stages"]), "modules": top(profile["modules"])}


def _flags(report: dict, summary: dict) -> list[str]:
    """The flags: an unverified replay, stale builds, low baseline coverage, frequent fallbacks, significantly worse
    steps, too few clusters, crops and misoriented reads, then the Holm-family note; at most ``MAX_FLAGS``."""
    config, suites = report["config"], report["suites"]
    flags = []
    if not summary["replay_verified"]:
        flags.append("Unverified replay: the AlignAIR code the replay mirrors changed since it was pinned ("
                     f"{', '.join(summary['replay_changes']) or 'unknown'}); re-pin (diagnostics/pin.py) before "
                     "trusting these numbers.")
    stale = config.get("stale_builds") or {}
    if stale:
        flags.append(_listed("Stale builds, diagnosed as built (rebuild to use this checkout's suite and truth "
                             "code)", [f"{name} ({', '.join(changes)})" for name, changes in stale.items()]))
    low = [f"{name} {metric} {_percent(entry['coverage'])}" for name, suite in suites.items()
           for metric in (*SUMMARY_METRICS, *(f"{g}.top1_in_set" for g in _GENES))
           if (entry := _scores(suite, "baseline").get(metric)) and entry.get("coverage") is not None
           and entry["coverage"] < COVERAGE_FLAG]
    if low:
        flags.append(_listed(f"Baseline coverage below {_percent(COVERAGE_FLAG, 0)}", low))
    frequent = []
    for name, suite in suites.items():
        diagnosed = suite["reads"]["diagnosed"] or 0
        worst: dict[str, tuple[int, str]] = {}
        for variant, result in (suite.get("variants") or {}).items():
            for reason, count in (result.get("fallbacks") or {}).items():
                if count > worst.get(reason, (0, ""))[0]:
                    worst[reason] = (count, variant)
        frequent += [f"{name} {reason} {_percent(count / diagnosed)} ({variant})"
                     for reason, (count, variant) in worst.items() if diagnosed and count / diagnosed > FALLBACK_FLAG]
    if frequent:
        flags.append(_listed(f"Fallbacks above {_percent(FALLBACK_FLAG, 0)} of reads (the truth could not be fed "
                             "in; see the suite reports)", frequent))
    worse = [f"{name} {comparison['from']} → {comparison['to']} {metric} "
             f"{format_value(metric, entry['delta'], delta=True)}"
             for name, suite in suites.items()
             for comparison in [*(suite.get("contributions") or []), *(suite.get("swaps") or [])]
             if comparison["to"] != "soft_decoder"   # the old decoder is expected to be worse
             for metric, entry in comparison["metrics"].items() if entry.get("verdict") == "from_better"]
    if worse:
        flags.append(_listed("Significantly worse with the truth fed in, or with the swap (not soft_decoder, the old decoder)", worse))
    few = []
    for name, suite in suites.items():
        entries = [entry for comparison in [*(suite.get("contributions") or []), *(suite.get("swaps") or [])]
                   for entry in comparison["metrics"].values() if entry.get("test") == "too_few_clusters"]
        if entries:
            clusters = sorted({entry["n_clusters"] for entry in entries})
            few.append(f"{name} ({len(entries)} comparisons, {_span(clusters)} clusters)")
    if few:
        flags.append(_listed("Too few clusters for any verdict (fewer than 10 clones or pairs; S9 needs scale ≥ 0.2)",
                             few))
    unseen = [f"{name} {suite['reads'][key]} {label}" for name, suite in suites.items()
              for key, label in (("cropped", "cropped"), ("misoriented", "misoriented"))
              if suite["reads"].get(key)]
    if unseen:
        flags.append(_listed("Reads whose boundaries are partly or not scored (cropped: past the cut; misoriented: "
                             "the model's frame is not the truth's)", unseen))
    return [*flags[:MAX_FLAGS - 1], HOLM_NOTE]


def _listed(head: str, items: list[str]) -> str:
    shown = items[:FLAG_ITEMS]
    more = f"; and {len(items) - FLAG_ITEMS} more" if len(items) > FLAG_ITEMS else ""
    return f"{head}: {'; '.join(shown)}{more}."


def _span(values: list) -> str:
    return str(values[0]) if values[0] == values[-1] else f"{values[0]}–{values[-1]}"


def diagnostics_markdown(report: dict) -> str:
    """The summary page, ``diagnostics.md`` (see the module docstring), linking each suite's ``<S>/report.md`` and
    the ``profile.md``."""
    summary = report["summary"]
    lines = [f"# AlignAIR diagnostics: model {summary['model_key']}", "", *_header(report, summary), ""]
    if summary.get("against"):
        lines += ["", f"## {AGAINST_SECTION}", "", *against_section(summary)]
    sections = (_errors_section, _calls_section, _boundaries_section, _metadata_section, _speed_section,
                _flags_section)
    for title, render in zip(SUMMARY_SECTIONS, sections):
        lines += ["", f"## {title}", "", *render(summary)]
    return "\n".join(lines) + "\n"


def _header(report: dict, summary: dict) -> list[str]:
    config = report["config"]
    reads = summary["reads"]
    max_cases = config.get("max_cases")
    seconds = summary.get("seconds")
    clustered = summary["clustered"]
    parts = [
        f"- Model `{summary['model_key']}`{labelled(summary.get('label'))} ({summary['source']}) on "
        f"{summary['device']}, batch size "
        f"{summary['batch_size']}: {len(reads)} suites × {sum(reads.values())} reads on the {summary['split']} split "
        f"({', '.join(f'{name} {n}' for name, n in reads.items())}; max cases {max_cases or 'all'}).",
        f"- Computed {', '.join(summary['computed']) or 'none'}; reused {', '.join(summary['reused']) or 'none'}"
        f" from the cache. Wall time {MISSING if seconds is None else f'{seconds:.1f} s'}.",
        "- Cluster-aware comparisons: " + (
            ", ".join(f"{name} ({entry['clusters']} {'/'.join(entry['keys'])} clusters)"
                      for name, entry in clustered.items()) + "; every other suite is compared read by read."
            if clustered else "none (no suite has clones or orientation pairs); every comparison is read by read."),
    ]
    links = [f"[{name}]({name}/report.md)" for name in reads]
    detail = f"Per-suite detail: {', '.join(links)}."
    if report.get("profile"):
        detail += " Speed profile: [profile.md](profile.md)."
    return [*parts, "", detail]


def _errors_section(summary: dict) -> list[str]:
    lines = [_ERRORS_NOTE]
    for metric, rows in summary["errors"].items():
        lines += ["", f"### {metric}", ""]
        if not rows:
            lines.append("No suite scored this metric.")
            continue
        table = []
        for name, row in rows.items():
            gap = MISSING if row["true_germline"] is None else (
                f"{format_value(metric, row['true_germline'])} (gap {100 * row['gap']:.1f} pp)")
            table.append([name, format_cell(metric, row["baseline"]),
                          *(_step_cell(metric, row["steps"].get(stage)) for stage in STAGES), gap,
                          _step_cell(metric, row["soft_decoder"]), _paired_span(row)])
        lines += markdown_table(["Suite", "Baseline", *(f"Δ {STAGE_LABELS[stage]}" for stage in STAGES),
                                 "true_germline (gap)", "Δ soft_decoder", "n paired"], table)
    return lines


def _step_cell(metric: str, entry: dict | None) -> str:
    """A Δ in percentage points, bold when significant, ``(worse)`` when significantly worse, with the stages a step
    spans when it spans more than one (a variant was not run)."""
    if not entry or entry.get("delta") is None:
        return MISSING
    text = format_value(metric, entry["delta"], delta=True)
    verdict = entry.get("verdict")
    if verdict in ("to_better", "from_better"):
        text = f"**{text}**" + (" (worse)" if verdict == "from_better" else "")
    stages = entry.get("stages") or []
    if len(stages) > 1:
        text += f" ({' + '.join(STAGE_LABELS.get(stage, stage) for stage in stages)})"
    return text


def _paired_span(row: dict) -> str:
    n, k = row["n_paired"], row["n_clusters"]
    if not n:
        return MISSING
    text = _span(n)
    return f"{text} ({_span(k)} clusters)" if k and k[1] < n[0] else text


def _calls_section(summary: dict) -> list[str]:
    rows = []
    for name, genes in summary["calls"].items():
        def cells(render) -> str:
            return " / ".join(render(g, genes[g]) if g in genes else MISSING for g in _GENES)

        rows.append([name,
                     cells(lambda g, c: _points(c["top1_in_set"])),
                     cells(lambda g, c: _step_cell(f"{g}.top1_in_set", c["true_calls"]).replace(" pp", "")),
                     cells(lambda g, c: MISSING if c["head.allele_top1"] is None else
                           f"{_points(c['head.allele_top1'])}→{_points(c['head.allele_top3'])}"),
                     cells(lambda g, c: MISSING if c["ece"] is None else f"{c['ece']:.3f}")])
    if not rows:
        return ["No allele call was scored."]
    return [_CALLS_NOTE, "", *markdown_table(["Suite", "Top-1 in set, % (V / D / J)", "Δ true calls, pp",
                                              "Head top-1 → top-3, %", "ECE"], rows)]


def _points(value) -> str:
    """A rate in percent, one decimal, without the sign."""
    return MISSING if value is None else f"{100 * value:.1f}"


def _boundaries_section(summary: dict) -> list[str]:
    columns = [(g, edge) for g in _GENES for edge in _EDGES]
    rows = [[name, *(_jump(edges[f"head.{g}.{edge}_exact"]) for g, edge in columns)]
            for name, edges in summary["boundaries"].items()]
    return [_BOUNDARIES_NOTE, "", *markdown_table(["Suite", *(f"{g.upper()} {edge}" for g, edge in columns)], rows)]


def _jump(rates: dict) -> str:
    soft, argmax = rates.get("soft"), rates.get("argmax")
    if soft is None or argmax is None:
        return MISSING
    return f"{round(100 * soft)}→{round(100 * argmax)}"


def _metadata_section(summary: dict) -> list[str]:
    columns = dict(METADATA_HEADS)
    if any(heads.get(head) is not None for heads in summary["metadata"].values() for head in D_PRESENCE_HEADS.values()):
        columns.update(D_PRESENCE_HEADS)
    rows = []
    for name, heads in summary["metadata"].items():
        cells = []
        for head in columns.values():
            value = heads.get(head)
            cells.append(MISSING if value is None else f"{value:.3f}" if "_err" in head else _percent(value))
        rows.append([name, *cells])
    return markdown_table(["Suite", *columns], rows)


def _speed_section(summary: dict) -> list[str]:
    speed = summary.get("speed")
    if not speed:
        return ["Not measured: the profile was turned off (`--no-profile`, or `profile=False`; an in-memory model is "
                "profiled only with `profile=True`)."]
    reused = "; reused from the cache" if speed["reused"] else ""
    return [f"{speed['reads_per_second']:.1f} reads/s on {speed['device']} (batch size {speed['batch_size']}, "
            f"{speed['reads']} reads{reused}).",
            "Top stages: " + ", ".join(f"{name} {_percent(share)}" for name, share in speed["stages"])
            + "; top modules (share of the forward stage): "
            + ", ".join(f"{name} {_percent(share)}" for name, share in speed["modules"]) + "."]


def _flags_section(summary: dict) -> list[str]:
    return [f"- {flag}" for flag in summary["flags"]]


# --- one suite's detail -----------------------------------------------------------------------------------------

def suite_markdown(report: dict, name: str) -> str:
    """Suite ``name``'s detail, ``<S>/report.md`` (see the module docstring), linking back to the summary."""
    model, config, suite = report["model"], report["config"], report["suites"][name]
    lines = [f"# AlignAIR diagnostics: model {model['model_key']}, suite {name}", "",
             f"Back to the [summary](../diagnostics.md). {_model_text(model, config)}", ""]
    if suite.get("replay_verified") is False:
        lines += [_unverified_text(suite.get("replay_changes") or {}), ""]
    reads = suite["reads"]
    clusters = suite.get("clusters") or {}
    lines += markdown_table(["Built", "Diagnosed", "Scored", "Excluded", "Cropped", "Misoriented", "Clusters"],
                            [[*(str(reads[key]) for key in _READS), str(reads.get("misoriented", MISSING)),
                              format_clusters(clusters)]])
    seconds = f"computed in {suite['seconds']:.1f} s"
    lines += ["", f"Result {'reused from the cache (' + seconds + ')' if suite.get('reused') else seconds}."]
    sections = ((_HEADS_NOTE, _suite_heads), (_with_cluster_note(suite, "contributions", _CONTRIBUTIONS_NOTE),
                                              _suite_contributions),
                (_CALIBRATION_NOTE, _suite_calibration), (_with_cluster_note(suite, "swaps", _SWAPS_NOTE),
                                                          _suite_swaps),
                (_FALLBACKS_NOTE, _suite_fallbacks))
    for title, (note, render) in zip(SUITE_SECTIONS, sections):
        lines += ["", f"## {title}", "", note, "", *render(suite)]
    return "\n".join(lines) + "\n"


def _model_text(model: dict, config: dict) -> str:
    max_cases = config.get("max_cases")
    return (f"Model key `{model['model_key']}`: weights SHA-256 `{model['weights_sha256'][:12]}` "
            f"({model['source']}), AlignAIR code SHA-256 `{model['code_sha256'][:12]}` ({model['alignair_source']}), "
            f"torch {model['torch_version']}. Run on {config['device']} with batch size {config['batch_size']}, "
            f"max cases {max_cases if max_cases else 'all'} per suite, variants "
            f"{', '.join(config['variants']) or 'none'}; run directory {config['run_dir']}. Every suite is on the "
            "dev split. A suite's result is cached under the model key and reused only while the model's weights, "
            "the AlignAIR code, torch, the assay's scoring code, the suite's build, the replay pin and the "
            "configuration (variants, max cases, device, batch size) are all unchanged.")


def _unverified_text(changes) -> str:
    names = ", ".join(f"`{name}`" for name in changes) or "unknown"
    return (f"**Unverified replay.** The AlignAIR code the replay mirrors has changed since it was pinned ({names}), "
            "and these diagnostics were run anyway (`allow_unverified_replay`): the replay may not be the product "
            "pipeline. Re-pin it (`diagnostics/pin.py`) before trusting them.")


def _with_cluster_note(suite: dict, field: str, note: str) -> str:
    """``note``, followed by ``CLUSTER_NOTE`` when a comparison under ``field`` was tested by cluster."""
    clustered = any(is_clustered(entry) for comparison in suite.get(field) or []
                    for entry in comparison["metrics"].values())
    return f"{note}\n\n{CLUSTER_NOTE}" if clustered else note


def _suite_heads(suite: dict) -> list[str]:
    heads = suite.get("heads") or {}
    lines: list[str] = []
    rows = []
    for g in _GENES:
        for edge in _EDGES:
            names = [f"head.{g}.{edge}_{kind}.{decoder}" for kind, decoder in _BOUNDARY_COLUMNS]
            if any(name in heads for name in names):
                rows.append([f"{g.upper()} {edge}", *(format_cell(name, heads.get(name)) for name in names)])
    if rows:
        lines += ["Boundaries:", "", *markdown_table(
            ["Edge", "Exact (soft)", "Exact (argmax)", "Abs. error, nt (soft)", "Abs. error, nt (argmax)",
             "Signed error, nt (soft)", "Signed error, nt (argmax)"], rows), ""]
    rows = []
    for g in _GENES:
        names = [f"head.{g}.{metric}" for metric in _ALLELE_METRICS]
        if any(name in heads for name in names):
            rows.append([g.upper(), *(format_cell(name, heads.get(name)) for name in names)])
    if rows:
        lines += ["Alleles (ranked by calibrated probability):", "", *markdown_table(
            ["Gene", "Top-1 in set", "Top-3 in set", "Top-5 in set", "Top-1 generative (informational)",
             "Alleles at p ≥ 0.5 (informational)"], rows), ""]
    rows = [[name, format_cell(name, heads[name])] for name in _META_METRICS if name in heads]
    if rows:
        lines += ["Metadata heads:", "", *markdown_table(["Head metric", "Value"], rows)]
    confusion = suite.get("d_presence")
    if confusion:
        classes = confusion["classes"]
        callable_reads = sum(confusion["counts"][classes.index("callable")])
        lines += ["", "D presence, truth class (rows) × the head's class (columns), in reads:", "",
                  *markdown_table(["Truth \\ head", *classes],
                                  [[truth, *(str(count) for count in row)]
                                   for truth, row in zip(classes, confusion["counts"])]),
                  "", f"{confusion['callable_gated']} of the {callable_reads} truth-callable reads were gated away "
                      "(called absent or short: their D call dropped)."]
    return lines or ["No head metric applies to these reads."]


def _suite_contributions(suite: dict) -> list[str]:
    steps = suite.get("contributions") or []
    if not steps:
        return ["No cumulative-chain variant was run (beyond the baseline), so there is no step to show."]
    metrics = _metrics(suite, steps)
    rows = [[name, *(_delta_cell(name, step["metrics"].get(name)) for step in steps)] for name in metrics]
    lines = markdown_table(["Metric", *(f"Δ {', '.join(step['stages'])}" for step in steps)], rows)
    for step in steps:
        lines += ["", f"### {step['from']} → {step['to']} ({', '.join(step['stages'])})", "",
                  *_comparison_table(step, metrics)]
    return lines


def _suite_swaps(suite: dict) -> list[str]:
    swaps = suite.get("swaps") or []
    if not swaps:
        return ["No single-swap variant was run."]
    metrics = _metrics(suite, swaps)
    lines: list[str] = []
    for swap in swaps:
        lines += [*([""] if lines else []), f"### {swap['from']} → {swap['to']}", "",
                  *_comparison_table(swap, metrics)]
    return lines


def _metrics(suite: dict, comparisons: list[dict]) -> list[str]:
    """The compared metrics in the suite's family order, then any other a comparison has."""
    present = dict.fromkeys(name for comparison in comparisons for name in comparison["metrics"])
    return [name for name in suite.get("metrics") or [] if name in present] + [
        name for name in present if name not in (suite.get("metrics") or [])]


def _verdict(entry: dict) -> str:
    """The verdict from the "to" variant's side, noting a comparison not tested for lack of clusters."""
    verdict = VERDICTS.get(entry["verdict"], entry["verdict"])
    return f"{verdict}, too few clusters" if entry.get("test") == "too_few_clusters" else verdict


def _delta_cell(name: str, entry: dict | None) -> str:
    """Δ with its verdict and its paired count (``n=1130 in 5 clusters`` when tested by cluster)."""
    if entry is None or entry.get("delta") is None:
        return MISSING
    return (f"{format_value(name, entry['delta'], delta=True)} ({_verdict(entry)}; "
            f"n={format_paired(entry, inline=True)})")


def _comparison_table(comparison: dict, metrics: list[str]) -> list[str]:
    a, b = comparison["from"], comparison["to"]
    rows = []
    for name in metrics:
        e = comparison["metrics"].get(name)
        if e is None:
            continue
        own_a = {"mean": e["mean_from"], "n": e["n_from"], "coverage": e["coverage_from"]}
        own_b = {"mean": e["mean_to"], "n": e["n_to"], "coverage": e["coverage_to"]}
        p = e.get("p_value")
        verdict = VERDICTS.get(e["verdict"], e["verdict"])
        if e.get("test") == "too_few_clusters":
            verdict += " (too few clusters)"
        rows.append([name, format_cell(name, own_a), format_cell(name, own_b), format_paired(e),
                     format_value(name, e.get("mean_from_paired")), format_value(name, e.get("mean_to_paired")),
                     format_value(name, e.get("delta"), delta=True), format_ci(name, e.get("ci95")),
                     MISSING if p is None else f"{p:.3g}", verdict])
    return markdown_table(["Metric", a, b, "n paired", f"{a} (paired)", f"{b} (paired)", "Δ", "95% CI", "p",
                           "Verdict"], rows)


def _suite_calibration(suite: dict) -> list[str]:
    tables = suite.get("calibration") or {}
    genes = [g for g in _GENES if g in tables]
    if not genes:
        return ["No allele head was calibrated on these reads."]
    rows = [[g.upper(), str(tables[g]["n"]), _percent(tables[g]["confidence"]), _percent(tables[g]["accuracy"]),
             MISSING if tables[g]["ece"] is None else f"{tables[g]['ece']:.3f}"] for g in genes]
    lines = markdown_table(["Gene", "n", "Mean confidence", "Accuracy", "ECE"], rows)
    edges = tables[genes[0]]["bins"]
    rows = []
    for k, edge in enumerate(edges):
        cells = [tables[g]["bins"][k] for g in genes]
        if any(cell["n"] for cell in cells):
            rows.append([f"{edge['lo']:.1f}–{edge['hi']:.1f}",
                         *(f"{cell['n']}: {_percent(cell['accuracy'])} / {_percent(cell['confidence'])}"
                           if cell["n"] else MISSING for cell in cells)])
    if rows:
        lines += ["", *markdown_table(["Top-1 probability", *(g.upper() for g in genes)], rows)]
    return lines


def _suite_fallbacks(suite: dict) -> list[str]:
    rows = [[variant, reason, str(count)] for variant, result in (suite.get("variants") or {}).items()
            for reason, count in (result.get("fallbacks") or {}).items()]
    if not rows:
        return ["No variant fell back to the model's output."]
    return markdown_table(["Variant", "Fallback", "Reads"], rows)


# --- one suite's comparison with another model ------------------------------------------------------------------

# --- the profile ------------------------------------------------------------------------------------------------

def profile_markdown(report: dict) -> str:
    """The speed profile's detail, ``profile.md``: time per stage and per network module."""
    profile = report.get("profile")
    lines = [f"# AlignAIR diagnostics: model {report['model']['model_key']}, speed profile", "",
             "Back to the [summary](diagnostics.md).", ""]
    if not profile:
        lines.append("Not measured: the profile was turned off.")
        return "\n".join(lines) + "\n"
    reused = "; reused from the cache" if profile.get("reused") else ""
    lines += [f"Device {profile['device']}, batch size {profile['batch_size']}, {profile['reads']} reads: "
              f"{profile['reads_per_second']:.1f} reads/s ({profile['seconds_per_run']:.2f} s per run, the median "
              f"of {profile['repeats']} runs{reused}).", "", _PROFILE_NOTE, ""]
    stages = [[name, f"{entry['ms_per_read']:.3f}", _percent(entry["share"])]
              for name, entry in profile["stages"].items()]
    lines += markdown_table(["Stage", "ms/read", "Share"], stages)
    modules = sorted(profile["modules"].items(), key=lambda item: (-item[1]["share"], item[0]))
    lines += ["", *markdown_table(["Module", "ms/read", "Share of forward"],
                                  [[name, f"{entry['ms_per_read']:.3f}", _percent(entry["share"])]
                                   for name, entry in modules])]
    return "\n".join(lines) + "\n"


def _percent(value, digits: int = 1) -> str:
    return MISSING if value is None else f"{100 * value:.{digits}f}%"
