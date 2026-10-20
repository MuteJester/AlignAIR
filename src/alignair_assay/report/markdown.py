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
MISSING = "—"


def _is_error(name: str) -> bool:
    return "_err" in name or "_errors" in name or "set_size" in name


def format_value(name: str, v, *, delta: bool = False) -> str:
    """A value of metric ``name``, or a difference (``delta``): rates as percentages and differences in percentage
    points, error-type metrics as raw numbers; None shows a dash."""
    if v is None:
        return MISSING
    if _is_error(name):
        return f"{v:+.3f}" if delta else f"{v:.3f}"
    return f"{100 * v:+.1f} pp" if delta else f"{100 * v:.1f}%"


def _cov(c) -> str:
    return MISSING if c is None else f"{100 * c:.0f}%"


def _order(names, headline: Sequence[str]) -> list[str]:
    names = set(names)
    return [n for n in headline if n in names] + sorted(names - set(headline))


def _esc(text: str) -> str:
    return str(text).replace("|", "\\|")


def markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    """The lines of a markdown table; ``|`` in a cell is escaped."""
    out = ["| " + " | ".join(_esc(h) for h in header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(_esc(c) for c in r) + " |" for r in rows]
    return out


def _mean(entry) -> float | None:
    return entry.get("mean") if entry else None


def format_cell(name: str, entry) -> str:
    """A summary entry (``mean``, ``n``, ``coverage``) as ``value (n=…, coverage%)``; a missing mean or zero
    coverage shows a dash."""
    mean = _mean(entry)
    if mean is None or not entry.get("coverage"):
        return MISSING
    return f"{format_value(name, mean)} (n={entry['n']}, {_cov(entry['coverage'])})"


def _score_md(report: dict, headline: Sequence[str]) -> list[str]:
    overall = report["overall"]
    lines = [f"# AlignAIR assay: {report['tool']}", "", "## Overall", ""]
    rows = [[n, format_value(n, _mean(overall[n])), str(overall[n]["n"]), _cov(overall[n]["coverage"])]
            for n in _order(overall, headline)]
    lines += markdown_table(["Metric", "Value", "n", "Coverage"], rows)
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
        n_paired = e.get("n_paired")
        rows.append([n, format_cell(n, e.get("a")), format_cell(n, e.get("b")),
                     MISSING if n_paired is None else str(n_paired),
                     format_value(n, e.get("mean_a_paired")), format_value(n, e.get("mean_b_paired")),
                     format_value(n, e.get("delta"), delta=True), _ci(n, e), e.get("verdict") or MISSING])
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
