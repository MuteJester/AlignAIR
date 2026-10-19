"""Markdown rendering of score and comparison reports."""
from __future__ import annotations

from alignair_assay.report.build import COMPARISON_KIND, SCORE_KIND

HEADLINE = ("clone_key.exact", "junction.nt_exact", "v.top1_in_set", "j.top1_in_set", "v.mutation_count_err",
            "d.top1_in_set", "productive.correct")
NOTE = "Verdicts: Holm-corrected paired bootstrap; Δ and CI are raw (b − a)."
MISSING = "—"


def _is_error(name: str) -> bool:
    return "_err" in name or "_errors" in name or "set_size" in name


def _val(name: str, v, *, delta: bool = False) -> str:
    if v is None:
        return MISSING
    if _is_error(name):
        return f"{v:+.3f}" if delta else f"{v:.3f}"
    return f"{100 * v:+.1f}%" if delta else f"{100 * v:.1f}%"


def _cov(c) -> str:
    return MISSING if c is None else f"{100 * c:.0f}%"


def _order(names) -> list[str]:
    names = set(names)
    return [n for n in HEADLINE if n in names] + sorted(names - set(HEADLINE))


def _esc(text: str) -> str:
    return str(text).replace("|", "\\|")


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(_esc(h) for h in header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(_esc(c) for c in r) + " |" for r in rows]
    return out


def _mean(entry) -> float | None:
    return entry.get("mean") if entry else None


def _cell(name: str, entry) -> str:
    """``value (coverage%)``; a missing mean or zero coverage shows a dash."""
    mean = _mean(entry)
    if mean is None or not entry.get("coverage"):
        return MISSING
    return f"{_val(name, mean)} ({_cov(entry['coverage'])})"


def _score_md(report: dict) -> list[str]:
    overall = report["overall"]
    lines = [f"# AlignAIR assay: {report['tool']}", "", "## Overall", ""]
    rows = [[n, _val(n, _mean(overall[n])), str(overall[n]["n"]), _cov(overall[n]["coverage"])]
            for n in _order(overall)]
    lines += _table(["Metric", "Value", "n", "Coverage"], rows)
    for dim, labels in report["by_slice"].items():
        lines += ["", f"## By {dim}", ""]
        names = [n for n in HEADLINE if any(n in m for m in labels.values())]
        rows = [[n] + [_cell(n, labels[lab].get(n)) for lab in labels] for n in names]
        lines += _table(["Metric"] + list(labels), rows)
    return lines


def _ci(name: str, entry) -> str:
    ci = entry.get("ci95")
    if not ci:
        return MISSING
    return f"[{_val(name, ci[0], delta=True)}, {_val(name, ci[1], delta=True)}]"


def _comparison_md(report: dict) -> list[str]:
    a, b = report["a"], report["b"]
    comp = report["comparison"]
    lines = [f"# AlignAIR assay: {a} vs {b}", "", "## Overall", ""]
    rows = []
    for n in _order(comp):
        e = comp[n]
        rows.append([n, _val(n, _mean(e["a"])), _val(n, _mean(e["b"])), _val(n, e["delta"], delta=True), _ci(n, e),
                     e.get("verdict") or MISSING, _cov(e["coverage_a"]), _cov(e["coverage_b"])])
    lines += _table(["Metric", a, b, "Δ (b−a)", "95% CI", "Verdict", f"Coverage {a}", f"Coverage {b}"], rows)
    lines += ["", NOTE]
    sa, sb = report["by_slice"]["a"], report["by_slice"]["b"]
    for dim in sorted(set(sa) | set(sb)):
        la, lb = sa.get(dim, {}), sb.get(dim, {})
        labels = sorted(set(la) | set(lb))
        lines += ["", f"## By {dim} ({a} / {b})", ""]
        names = [n for n in HEADLINE if any(n in m for m in list(la.values()) + list(lb.values()))]
        rows = [[n] + [f"{_cell(n, la.get(lab, {}).get(n))} / {_cell(n, lb.get(lab, {}).get(n))}"
                       for lab in labels] for n in names]
        lines += _table(["Metric"] + labels, rows)
    return lines


def to_markdown(report: dict) -> str:
    """Markdown summary of a score report or a comparison report."""
    kind = report.get("kind")
    if kind == SCORE_KIND:
        lines = _score_md(report)
    elif kind == COMPARISON_KIND:
        lines = _comparison_md(report)
    else:
        raise ValueError(f"unknown report kind: {kind!r}")
    return "\n".join(lines) + "\n"
