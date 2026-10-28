"""Markdown rendering of score and comparison reports, and the table and value formatting the run report shares.

The headline rows (default ``core.headline.HEADLINE``) are a parameter: a suite's report leads with its own primary
metrics, and its slice tables show only those.
"""
from __future__ import annotations

from collections.abc import Sequence

from alignair_assay.core.headline import HEADLINE
from alignair_assay.report.build import COMPARISON_KIND, SCORE_KIND

NOTE = ("Each tool's value is its mean over every case it covers (n, coverage). Δ (b − a) and its CI are computed on "
        "the paired subset (cases both tools cover), whose means are shown; for rates they are differences in "
        "percentage points (pp). Verdicts: Holm-corrected paired tests (exact McNemar for 0/1 metrics, paired z "
        "otherwise).")
# Shown under a comparison table that has a clustered row (``n_clusters`` below ``n_paired``).
CLUSTER_NOTE = ("n paired (K clusters): the reads of one clone, or the two strands of one read, are one cluster, "
                "and a row with clusters is tested on the per-cluster sums of the paired differences (clustered "
                "McNemar for 0/1 metrics, clustered z otherwise) with a cluster-robust CI, so a few clusters cannot "
                "carry a verdict. With fewer than 10 clusters there is no test (too few clusters: p = 1, tie).")
MISSING = "—"


def alignair_code_text(codes: dict) -> str:
    """The AlignAIR code two checkpoint variants ran under (tool -> its fingerprint's ``code_sha256``, None when not
    recorded): ``both ran under AlignAIR code `<first 12>``` for one code, each tool's otherwise, flagged."""
    digests = list(codes.values())
    if not digests or None in digests:
        return "AlignAIR code not recorded"
    if len(set(digests)) == 1:
        return f"both ran under AlignAIR code `{digests[0][:12]}`"
    return "different AlignAIR code: " + ", ".join(f"{tool} `{code[:12]}`" for tool, code in codes.items())


def _is_error(name: str) -> bool:
    """True for a metric shown as a raw number: errors and counts (not rates)."""
    return "_err" in name or "_errors" in name or "set_size" in name or "n_above_threshold" in name


def format_value(name: str, v, *, delta: bool = False) -> str:
    """A value of metric ``name``, or a difference (``delta``): rates as percentages and differences in percentage
    points, error-type metrics and counts as raw numbers; None shows a dash."""
    if v is None:
        return MISSING
    if _is_error(name):
        return f"{v:+.3f}" if delta else f"{v:.3f}"
    return f"{100 * v:+.1f} pp" if delta else f"{100 * v:.1f}%"


def _cov(c) -> str:
    return MISSING if c is None else f"{100 * c:.0f}%"


def _order(names, headline: Sequence[str]) -> list[str]:
    """The ``headline`` metrics present, in that order, then the others sorted; each ``{g}.top1_generative``
    (true-allele accuracy) right after its ``{g}.top1_in_set`` when both are there."""
    names = set(names)
    order = [n for n in headline if n in names] + sorted(names - set(headline))
    for name in [n for n in order if n.endswith(".top1_generative")]:
        twin = name.removesuffix(".top1_generative") + ".top1_in_set"
        if twin in names:
            order.remove(name)
            order.insert(order.index(twin) + 1, name)
    return order


def _esc(text: str) -> str:
    return str(text).replace("|", "\\|")


def markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    """The lines of a markdown table; ``|`` in a cell is escaped."""
    out = ["| " + " | ".join(_esc(h) for h in header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(_esc(c) for c in r) + " |" for r in rows]
    return out


def _mean(entry) -> float | None:
    return entry.get("mean") if entry else None


def is_clustered(entry) -> bool:
    """True for a comparison entry tested by cluster: fewer clusters than pairs."""
    n, k = entry.get("n_paired"), entry.get("n_clusters")
    return n is not None and k is not None and k < n


def format_paired(entry, *, inline: bool = False) -> str:
    """A comparison entry's paired count: ``n``, or ``n (K clusters)`` when it was tested by cluster
    (``n in K clusters`` with ``inline``); a dash without a count."""
    n = entry.get("n_paired")
    if n is None:
        return MISSING
    if not is_clustered(entry):
        return str(n)
    k = entry["n_clusters"]
    return f"{n} in {k} clusters" if inline else f"{n} ({k} clusters)"


def format_verdict(entry, verdict: str | None = None) -> str:
    """A comparison entry's verdict (``verdict`` when given, else its own), with ``(too few clusters)`` when it was
    not tested for lack of clusters; a dash without one."""
    text = verdict if verdict is not None else entry.get("verdict")
    if text is None:
        return MISSING
    return f"{text} (too few clusters)" if entry.get("test") == "too_few_clusters" else text


def format_cell(name: str, entry) -> str:
    """A summary entry (``mean``, ``n``, ``coverage``) as ``value (n=…, coverage%)``; a missing mean or zero
    coverage shows a dash."""
    mean = _mean(entry)
    if mean is None or not entry.get("coverage"):
        return MISSING
    return f"{format_value(name, mean)} (n={entry['n']}, {_cov(entry['coverage'])})"


def excess_note(min_reads) -> str:
    """The sentences explaining an excess-alleles table (``metrics.genotype.excess_alleles``) counted with
    ``min_reads``: what all calls and the avoidable calls count."""
    return (f"Excess alleles: of the donor genes with at least {min_reads} reads whose top-1 call is of that gene, "
            f"the share whose distinct top-1 alleles, each called on at least {min_reads} reads, outnumber the "
            "alleles the donor carries (none for a deleted gene); n is the number of those donor genes. All calls "
            "counts every top-1 call. Avoidable leaves out the defensible ones, a call in the read's truth set of "
            "an allele the donor does not carry (identical to a carried allele over the read, say): a "
            "sequence-only tool cannot avoid those, so only the avoidable share is a tool error.")


def format_excess(entry, *, avoidable: bool = False) -> str:
    """An excess-alleles entry (``n``, ``fraction``, ``avoidable_fraction``) as ``fraction% (n=…)``, the avoidable
    fraction when ``avoidable``; a missing entry shows a dash, and an entry over no donor gene (or without the
    fraction) a dash with its n."""
    if entry is None:
        return MISSING
    fraction = entry.get("avoidable_fraction" if avoidable else "fraction")
    return f"{MISSING if fraction is None else f'{100 * fraction:.1f}%'} (n={entry['n']})"


def _score_md(report: dict, headline: Sequence[str]) -> list[str]:
    overall = report["overall"]
    lines = [f"# AlignAIR assay: {report['tool']}", "", "## Overall", ""]
    rows = [[n, format_value(n, _mean(overall[n])), str(overall[n]["n"]), _cov(overall[n]["coverage"])]
            for n in _order(overall, headline)]
    lines += markdown_table(["Metric", "Value", "n", "Coverage"], rows)
    excess = report.get("excess_alleles")
    if excess is not None:
        genes = [g for g in ("v", "j") if g in excess]
        rows = [[label, *(format_excess(excess[g], avoidable=avoidable) for g in genes)]
                for label, avoidable in (("All calls", False), ("Avoidable", True))]
        lines += ["", "## Excess alleles", "", excess_note(excess["min_reads"]), "",
                  *markdown_table(["Calls", *(g.upper() for g in genes)], rows)]
    for dim, labels in report["by_slice"].items():
        lines += ["", f"## By {dim}", ""]
        names = [n for n in headline if any(n in m for m in labels.values())]
        rows = [[n] + [format_cell(n, labels[lab].get(n)) for lab in labels] for n in names]
        lines += markdown_table(["Metric"] + list(labels), rows)
    return lines


def _ci(name: str, entry) -> str:
    ci = entry.get("ci95")
    if not ci:
        return MISSING
    return f"[{format_value(name, ci[0], delta=True)}, {format_value(name, ci[1], delta=True)}]"


def _comparison_table(report: dict, names) -> list[str]:
    a, b = report["a"], report["b"]
    comp = report["comparison"]
    rows = []
    for n in names:
        e = comp[n]
        rows.append([n, format_cell(n, e.get("a")), format_cell(n, e.get("b")), format_paired(e),
                     format_value(n, e.get("mean_a_paired")), format_value(n, e.get("mean_b_paired")),
                     format_value(n, e.get("delta"), delta=True), _ci(n, e), format_verdict(e)])
    return markdown_table(["Metric", a, b, "n paired", f"{a} (paired)", f"{b} (paired)", "Δ (b−a)", "95% CI",
                           "Verdict"], rows)


def headline_table(report: dict, headline: Sequence[str] = HEADLINE) -> str:
    """The rows of a comparison report's overall table for the ``headline`` metrics it has, in that order."""
    names = [n for n in headline if n in report["comparison"]]
    return "\n".join(_comparison_table(report, names)) + "\n"


def _comparison_md(report: dict, headline: Sequence[str]) -> list[str]:
    a, b = report["a"], report["b"]
    lines = [f"# AlignAIR assay: {a} vs {b}", "", "## Overall", ""]
    lines += _comparison_table(report, _order(report["comparison"], headline))
    lines += ["", NOTE]
    if any(is_clustered(entry) for entry in report["comparison"].values()):
        lines += ["", CLUSTER_NOTE]
    informational = report.get("informational") or {}
    if informational:
        rows = [[n, format_cell(n, informational[n].get("a")), format_cell(n, informational[n].get("b"))]
                for n in _order(informational, headline)]
        lines += ["", "## Informational (not compared)", "",
                  "Reported for each tool over its own cases (n, coverage), but not tested: these metrics are "
                  "outside the Holm family (`metrics.INFORMATIONAL`).", "", *markdown_table(["Metric", a, b], rows)]
    if report.get("changes"):
        lines += ["", *_changes_md(a, b, report["changes"])]
    sa, sb = report["by_slice"]["a"], report["by_slice"]["b"]
    for dim in sorted(set(sa) | set(sb)):
        la, lb = sa.get(dim, {}), sb.get(dim, {})
        labels = sorted(set(la) | set(lb))
        lines += ["", f"## By {dim} ({a} / {b})", ""]
        names = [n for n in headline if any(n in m for m in list(la.values()) + list(lb.values()))]
        rows = [[n] + [f"{format_cell(n, la.get(lab, {}).get(n))} / {format_cell(n, lb.get(lab, {}).get(n))}"
                       for lab in labels] for n in names]
        lines += markdown_table(["Metric"] + labels, rows)
    return lines


def _changes_md(a: str, b: str, changes: dict) -> list[str]:
    """The reads ``b`` fixes (``a`` wrong, ``b`` right) and breaks on at least one 0/1 metric, with examples, per
    stratum and per metric (``stats.read_changes``)."""
    reads = changes["reads"]
    lines = ["## Reads fixed and broken", "",
             f"{b} fixes {reads['fixed']} reads {a} gets wrong and breaks {reads['broken']} it gets right, on at "
             "least one 0/1 metric (examples evenly spaced over the sorted ids)."]
    for side in ("fixed", "broken"):
        if reads.get(f"{side}_examples"):
            lines.append(f"- {side.capitalize()}, e.g. {', '.join(reads[f'{side}_examples'])}.")
    groups = reads.get("groups") or {}
    if groups:
        lines += ["", *markdown_table(["Stratum", "Fixed", "Broken"],
                                      [[label, str(n["fixed"]), str(n["broken"])] for label, n in groups.items()])]
    rows = [[metric, str(found["n_paired"]), str(found["fixed"]), str(found["broken"]),
             ", ".join(found["fixed_examples"]) or MISSING, ", ".join(found["broken_examples"]) or MISSING]
            for metric, found in changes["metrics"].items() if found["fixed"] or found["broken"]]
    if rows:
        lines += ["", *markdown_table(["Metric", "n paired", "Fixed", "Broken", "Fixed (examples)",
                                       "Broken (examples)"], rows)]
    return lines


def to_markdown(report: dict, headline: Sequence[str] = HEADLINE) -> str:
    """Markdown summary of a score report or a comparison report. The ``headline`` metrics lead the overall table,
    in that order, and are the rows of the slice tables."""
    kind = report.get("kind")
    if kind == SCORE_KIND:
        lines = _score_md(report, headline)
    elif kind == COMPARISON_KIND:
        lines = _comparison_md(report, headline)
    else:
        raise ValueError(f"unknown report kind: {kind!r}")
    return "\n".join(lines) + "\n"


def format_ci(name: str, ci) -> str:
    """A 95% CI ``[low, high]`` of metric ``name`` as a Δ (``format_value``); ``MISSING`` without one."""
    if not ci:
        return MISSING
    return f"[{format_value(name, ci[0], delta=True)}, {format_value(name, ci[1], delta=True)}]"


def format_clusters(clusters: dict) -> str:
    """A suite's ``clusters`` record as its detail table shows it: ``none`` without a cluster of two or more reads
    (``workflow.diagnose.cluster_counts`` counts only those), else the count with its keys and reads."""
    if not clusters.get("clusters"):
        return "none"
    return f"{clusters['clusters']} ({'/'.join(clusters['keys'])}; {clusters['cases']} reads)"
