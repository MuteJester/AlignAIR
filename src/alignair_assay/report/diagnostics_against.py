"""The diagnostics report's comparison with a baseline model (``diagnose --against``), from the plain dicts
``workflow.diagnose.compare_models`` and ``report.diagnostics`` build.

- ``against_summary``: the comparison's one-page summary, held in the report's ``summary`` (Δ = this model − the
  baseline; fixed: reads the baseline gets wrong and this model right), with the AlignAIR code each model ran under.
- ``against_section``: the summary page's ``AGAINST_SECTION``, first on ``diagnostics.md``: per suite the paired
  deltas of ``AGAINST_HEADLINE`` (the baseline's and this model's paired means, Δ, its 95% CI, the verdict and the
  reads fixed and broken), the boundary exact rates' Δ under both decoders (``AGAINST_BOUNDARIES``), and the reads
  fixed and broken per family with their totals.
- ``against_markdown``: one suite's detail of the comparison, ``<S>/against_<other key>.md``: ``AGAINST_SECTIONS``
  (the end-to-end and the head metrics, each model's own mean beside the paired means, Δ, CI, p and verdict; then
  the reads fixed and broken per 0/1 metric and per stratum, with example case ids).
"""
from __future__ import annotations

from alignair_assay.report.markdown import (
    CLUSTER_NOTE,
    MISSING,
    format_cell,
    format_ci,
    format_clusters,
    format_paired,
    format_value,
    labelled,
    markdown_table,
)

_GENES = ("v", "d", "j")
_EDGES = ("start", "end")
# The comparison with another model: its summary section (first on the summary page) and its per-suite detail.
AGAINST_SECTION = "Checkpoint comparison"
AGAINST_SECTIONS = ("End-to-end metrics", "Head metrics", "Reads fixed and broken")
# The compared families, end-to-end first, and their names in the report.
AGAINST_FAMILIES = ("baseline", "heads")
FAMILY_LABELS = {"baseline": "end-to-end", "heads": "heads"}
# The summary's headline metrics (the end-to-end ones, then the orientation head) and its boundary exact rates (both
# decoders).
AGAINST_HEADLINE = ("clone_key.exact", "junction.nt_exact", "v.top1_in_set", "d.top1_in_set", "j.top1_in_set",
                    "head.orientation.correct")
AGAINST_BOUNDARIES = tuple(f"head.{g}.{edge}_exact.{decoder}" for g in ("v", "d", "j") for edge in ("start", "end")
                           for decoder in ("soft", "argmax"))
# A comparison's verdict (``stats.compare``'s a is the baseline, b this model), and a metric no read pairs (the head
# boundaries pair only the reads both models orient as the truth): never a tie.
AGAINST_VERDICTS = {"b_better": "this better", "a_better": "baseline better", "tie_or_inconclusive": "tie"}
NOT_PAIRED = "not paired"
_AGAINST_NOTE = ("This model against the baseline (`--against`), on the same reads (the same suite builds and "
                 "settings). Δ = this model − the baseline over the reads both score (percentage points for rates), "
                 "with its 95% CI; fixed: reads the baseline gets wrong and this model right (a 0/1 metric), broken: "
                 "reads the baseline gets right and this model wrong. Verdicts: Holm within each suite's end-to-end "
                 "and head families, cluster-aware where a suite has clusters.")
_AGAINST_BOUNDARIES_NOTE = ("Boundary exact rates (head metrics): Δ (this − baseline) under the soft-argmax "
                            "decoder / the position logits' argmax (the product's decoder), in pp; bold: significant. Their CIs are in the "
                            "suite detail.")
_AGAINST_DETAIL_NOTE = ("Each model's own mean over its own reads (n, coverage), then the paired means over the reads "
                        "both score, whose difference is Δ (this − baseline), its 95% CI, p and the Holm-corrected "
                        "verdict within the family. Informational metrics are not compared.")


def against_summary(against: dict | None) -> dict | None:
    """The one-page summary of a comparison with a baseline model (``workflow.diagnose.compare_models``; None for
    None): the ``baseline``'s model key, its ``source``, ``same_model`` and the AlignAIR
    ``code`` each model was diagnosed under (``this``, ``baseline``: their code digests, None when not recorded, and
    ``same``: whether they are one code, None when either is unknown); per suite
    (``suites``) the ``cases``, the ``clusters`` of two or more reads, how many metrics each family ``compared``, the
    ``headline`` (each metric of ``AGAINST_HEADLINE`` and ``AGAINST_BOUNDARIES`` compared -> its ``family``, the
    paired means ``mean_baseline`` and ``mean_this``, ``delta`` (this − baseline), ``ci95``, ``verdict``, ``test``,
    ``p_value``, ``n_paired``, ``n_clusters`` and the reads ``fixed`` (the baseline wrong, this model right) and
    ``broken``, None for a metric that is not 0/1), ``deltas`` (every compared metric -> Δ), ``fixed`` and
    ``broken`` (every 0/1 metric -> reads), ``reads`` (family -> the reads fixed and broken on at least one of its
    metrics) and the ``significant`` metrics; and the ``totals`` (family -> reads fixed and broken, summed over the
    suites)."""
    if not against:
        return None
    suites = {}
    for name, suite in against["suites"].items():
        families = {family: suite[family] for family in AGAINST_FAMILIES if suite.get(family)}
        entries = {metric: (family, entry) for family, found in families.items()
                   for metric, entry in found["comparison"].items()}
        changes = {metric: change for found in families.values()
                   for metric, change in found["changes"]["metrics"].items()}
        headline = {}
        for metric in (*AGAINST_HEADLINE, *AGAINST_BOUNDARIES):
            if metric in entries:
                family, entry = entries[metric]
                headline[metric] = {"family": family, "mean_baseline": entry.get("mean_a_paired"),
                                    "mean_this": entry.get("mean_b_paired"),
                                    **{key: entry.get(key) for key in ("delta", "ci95", "verdict", "test", "p_value",
                                                                       "n_paired", "n_clusters")},
                                    "fixed": (changes.get(metric) or {}).get("fixed"),
                                    "broken": (changes.get(metric) or {}).get("broken")}
        suites[name] = {
            "cases": suite.get("cases"), "clusters": (suite.get("clusters") or {}).get("clusters", 0),
            "compared": {family: len(found["metrics"]) for family, found in families.items()},
            "headline": headline,
            "deltas": {metric: entry.get("delta") for metric, (_, entry) in entries.items()},
            "fixed": {metric: change["fixed"] for metric, change in changes.items()},
            "broken": {metric: change["broken"] for metric, change in changes.items()},
            "reads": {family: {"fixed": found["changes"]["reads"]["fixed"],
                               "broken": found["changes"]["reads"]["broken"]} for family, found in families.items()},
            "significant": [metric for metric, (_, entry) in entries.items()
                            if entry.get("verdict") not in (None, "tie_or_inconclusive")]}
    totals = {family: {kind: sum(suite["reads"][family][kind] for suite in suites.values() if family in suite["reads"])
                       for kind in ("fixed", "broken")}
              for family in AGAINST_FAMILIES if any(family in suite["reads"] for suite in suites.values())}
    code = against.get("code") or {}
    this, baseline = code.get("this"), code.get("baseline")
    same = None if None in (this, baseline) else this == baseline
    return {"baseline": against["baseline"], "source": against.get("source"),
            "same_model": bool(against.get("same_model")),
            "code": {"this": this, "baseline": baseline, "same": same}, "suites": suites, "totals": totals}


def against_section(summary: dict) -> list[str]:
    against = summary["against"]
    key = against["baseline"]
    lines = [f"Against the baseline `{key}` ({against['source'] or 'source not recorded'}). {_AGAINST_NOTE}", "",
             code_text(against.get("code"))]
    if against["same_model"]:
        lines += ["", "Both are the same model key, so this compares one record with itself: every Δ is 0. Compare "
                      "with the model in memory, or another file of the same weights (other bytes), to diagnose it "
                      "twice."]
    rows = []
    for name, suite in against["suites"].items():
        for metric in AGAINST_HEADLINE:
            entry = suite["headline"].get(metric)
            if entry:
                delta = format_value(metric, entry["delta"], delta=True) + _paired_n(entry, suite.get("cases"))
                rows.append([name, metric, format_value(metric, entry["mean_baseline"]),
                             format_value(metric, entry["mean_this"]), delta, format_ci(metric, entry["ci95"]),
                             _against_verdict(entry), _fixed_broken(entry)])
    header = ["Suite", "Metric", "Baseline", "This model", "Δ (this − baseline)", "95% CI", "Verdict", "Fixed / broken"]
    lines += ["", *(markdown_table(header, rows) if rows else
                    ["No end-to-end metric was compared (no variant was run)."])]
    columns = [(g, edge) for g in _GENES for edge in _EDGES]
    rows = [[name, *(_boundary_delta(suite["headline"], g, edge, suite.get("cases")) for g, edge in columns)]
            for name, suite in against["suites"].items()]
    lines += ["", _AGAINST_BOUNDARIES_NOTE, "",
              *markdown_table(["Suite", *(f"{g.upper()} {edge}" for g, edge in columns)], rows)]
    per_suite = "; ".join(f"{name}: " + ", ".join(f"{FAMILY_LABELS[family]} {reads['fixed']} / {reads['broken']}"
                                                  for family, reads in suite["reads"].items())
                          for name, suite in against["suites"].items())
    totals = ", ".join(f"{FAMILY_LABELS[family]} {reads['fixed']} / {reads['broken']}"
                       for family, reads in against["totals"].items())
    links = ", ".join(f"[{name}]({name}/against_{key}.md)" for name in against["suites"])
    lines += ["", f"Reads fixed / broken on at least one metric of the family: {per_suite}. "
                  f"Total: {totals or MISSING}.", "", f"Per-suite detail: {links}."]
    return lines


def code_text(code: dict | None) -> str:
    """The AlignAIR code each model of a comparison was diagnosed under (``against_summary``'s ``code``), as one line
    that flags a different code: every Δ then holds the code change too."""
    code = code or {}
    if code.get("same") is None:
        return "AlignAIR code not recorded."
    if code["same"]:
        return f"Both ran under AlignAIR code `{code['this'][:12]}`."
    return (f"**Different AlignAIR code**: this model ran under `{code['this'][:12]}`, the baseline under "
            f"`{code['baseline'][:12]}`, so every Δ holds the code change as well as the weights'.")


def _against_verdict(entry: dict) -> str:
    """The verdict from this model's side; ``NOT_PAIRED`` when no read pairs (never a tie)."""
    if entry.get("n_paired") == 0:
        return NOT_PAIRED
    verdict = AGAINST_VERDICTS.get(entry.get("verdict"), entry.get("verdict") or MISSING)
    return f"{verdict} (too few clusters)" if entry.get("test") == "too_few_clusters" else verdict


def _paired_n(entry: dict, cases) -> str:
    """`` (n=<paired>)`` when the metric pairs fewer reads than the suite's ``cases`` (none when it pairs them all,
    or the count is unknown)."""
    n = entry.get("n_paired")
    return f" (n={n})" if isinstance(n, int) and isinstance(cases, int) and n < cases else ""


def _fixed_broken(entry: dict) -> str:
    if entry.get("fixed") is None:
        return MISSING
    return f"{entry['fixed']} / {entry['broken']}"


def _boundary_delta(headline: dict, g: str, edge: str, cases) -> str:
    """One edge's Δ under each decoder, soft / argmax, in pp (bold when significant), with the paired n when below the
    suite's ``cases``; ``NOT_PAIRED`` when no read pairs."""
    cells = []
    for decoder in ("soft", "argmax"):
        entry = headline.get(f"head.{g}.{edge}_exact.{decoder}")
        if entry and entry.get("n_paired") == 0:
            cells.append(NOT_PAIRED)
            continue
        if not entry or entry.get("delta") is None:
            cells.append(MISSING)
            continue
        text = f"{100 * entry['delta']:+.1f}"
        text = f"**{text}**" if entry.get("verdict") in ("a_better", "b_better") else text
        cells.append(text + _paired_n(entry, cases))
    return " / ".join(cells)


def against_markdown(report: dict, name: str) -> str:
    """Suite ``name``'s detail of the comparison with the baseline, ``<S>/against_<baseline key>.md`` (see the
    module docstring), linking back to the summary."""
    model, against = report["model"], report["against"]
    suite = against["suites"][name]
    key = against["baseline"]
    clusters = suite.get("clusters") or {}
    lines = [f"# AlignAIR diagnostics: model {model['model_key']} against {key}, suite {name}", "",
             f"Back to the [summary](../diagnostics.md). This model: `{model['model_key']}`"
             f"{labelled(model.get('label'))} ({model['source']}); the baseline: `{key}` "
             f"({against.get('source') or 'source not recorded'}). {suite['cases']} diagnosed reads, the same for "
             f"both; clusters: {format_clusters(clusters)}. {_AGAINST_NOTE}", "",
             code_text(report["summary"]["against"]["code"])]
    if against.get("same_model"):
        lines += ["", "Both are the same model key: one record compared with itself."]
    if clusters.get("clusters"):
        lines += ["", CLUSTER_NOTE]
    for title, family in zip(AGAINST_SECTIONS[:2], AGAINST_FAMILIES):
        found = suite.get(family)
        lines += ["", f"## {title}", "", _AGAINST_DETAIL_NOTE, ""]
        lines += (_against_table(found) if found else
                  ["No variant was run, so no end-to-end score is compared." if family == "baseline" else
                   "No head metric applies to these reads."])
    lines += ["", f"## {AGAINST_SECTIONS[2]}", "", *_changes_tables(suite)]
    return "\n".join(lines) + "\n"


def _against_table(found: dict) -> list[str]:
    rows = []
    for metric in found["metrics"]:
        e = found["comparison"].get(metric)
        if e is None:
            continue
        p = e.get("p_value")
        # stats.compare's a is the baseline, b this model
        rows.append([metric, format_cell(metric, e.get("a")), format_cell(metric, e.get("b")), format_paired(e),
                     format_value(metric, e.get("mean_a_paired")), format_value(metric, e.get("mean_b_paired")),
                     format_value(metric, e.get("delta"), delta=True), format_ci(metric, e.get("ci95")),
                     MISSING if p is None else f"{p:.3g}", _against_verdict(e)])
    return markdown_table(["Metric", "Baseline", "This model", "n paired", "Baseline (paired)", "This (paired)",
                           "Δ (this − baseline)", "95% CI", "p", "Verdict"], rows)


def _changes_tables(suite: dict) -> list[str]:
    families = [(family, suite[family]) for family in AGAINST_FAMILIES if suite.get(family)]
    lines = ["Reads the baseline gets wrong and this model right (fixed), and the baseline right and this model "
             "wrong (broken), per 0/1 metric, with up to 10 case ids each; the metrics where no read changed are left "
             "out.", ""]
    totals = []
    rows = []
    for family, found in families:
        changes = found["changes"]
        totals.append(f"{FAMILY_LABELS[family]} {changes['reads']['fixed']} fixed / {changes['reads']['broken']} "
                      "broken")
        for metric, change in changes["metrics"].items():
            if change["fixed"] or change["broken"]:
                rows.append([FAMILY_LABELS[family], metric, str(change["n_paired"]), str(change["fixed"]),
                             str(change["broken"]), ", ".join(change["fixed_examples"]) or MISSING,
                             ", ".join(change["broken_examples"]) or MISSING])
    lines.append(f"Reads changed on at least one metric: {'; '.join(totals) or MISSING}.")
    strata = [[FAMILY_LABELS[family], label, str(counts["fixed"]), str(counts["broken"])]
              for family, found in families
              for label, counts in (found["changes"]["reads"].get("groups") or {}).items()]
    if strata:
        lines += ["", "Per stratum, the reads changed on at least one metric of the family:", "",
                  *markdown_table(["Family", "Stratum", "Fixed", "Broken"], strata)]
    if rows:
        lines += ["", *markdown_table(["Family", "Metric", "n paired", "Fixed", "Broken", "Fixed (examples)",
                                       "Broken (examples)"], rows)]
    else:
        lines += ["", "No read is fixed or broken on any 0/1 metric."]
    return lines
