"""A screen's headline: the candidate against the baseline at the same step, per suite, on one screen.

``render_headline`` takes the assay's comparison summary (``evaluate(..., against=...)``'s
``result.against["summary"]``, ``report.diagnostics_against.against_summary``) and renders three tables, one row
per suite and a tally row (suites where the candidate is better / worse / tied):

1. ``HEADLINE``: ``clone_key.exact``, ``junction.nt_exact`` and V / D / J top-1 in set, each as the Δ (candidate −
   baseline, percentage points), its 95% CI, the verdict and the reads fixed / broken;
2. and 3. ``BOUNDARIES``: each boundary's exact rate under the product's soft-argmax decoder, then under the
   position logits' argmax, as the Δ, the verdict and the reads fixed / broken. Their CIs are in the assay's
   per-suite detail (``<S>/against_<baseline key>.md``), as on the assay's own summary page.

Verdicts are the assay's (Holm within each suite's family, cluster-aware where a suite has clusters): ``+`` the
candidate better, ``-`` the baseline better, ``=`` a tie, ``?`` a tie with too few clusters to test. ``not paired``:
no read pairs; ``n/a``: the suite does not compare the metric. Plain text, no IO.
"""
from __future__ import annotations

from collections.abc import Sequence

HEADLINE = (("clone_key.exact", "clone_key.exact"), ("junction.nt_exact", "junction.nt_exact"), ("v.top1_in_set", "V top-1"),
            ("d.top1_in_set", "D top-1"), ("j.top1_in_set", "J top-1"))
BOUNDARIES = tuple((f"head.{gene}.{edge}_exact", f"{gene.upper()} {edge}") for gene in "vdj" for edge in ("start", "end"))
DECODERS = (("soft", "Boundary exact rates, the product's soft-argmax decoder:"),
            ("argmax", "Boundary exact rates, the position logits' argmax decoder:"))
MARKS = {"b_better": "+", "a_better": "-", "tie_or_inconclusive": "="}
LEGEND = ("Δ = candidate − baseline (pp) [95% CI], verdict (+ candidate better, - baseline better, = tie, ? too few "
          "clusters),", "reads fixed/broken (fixed: the baseline wrong, the candidate right); n/a: not compared.")
TALLY = "+/-/="
_SEPARATOR = "  "


def _mark(entry: dict) -> str:
    mark = MARKS.get(entry.get("verdict"), "?")
    return "?" if entry.get("test") == "too_few_clusters" and mark == "=" else mark


def _cell(entry: dict | None, *, ci: bool) -> str:
    if not entry or entry.get("delta") is None:
        return "n/a"
    if entry.get("n_paired") == 0:
        return "not paired"
    text = f"{100 * entry['delta']:+.1f}"
    if ci and entry.get("ci95"):
        low, high = entry["ci95"]
        text += f" [{100 * low:.1f},{100 * high:.1f}]"
    text += f" {_mark(entry)}"
    if entry.get("fixed") is not None:
        text += f" {entry['fixed']}/{entry['broken']}"
    return text


def _tally(entries: Sequence[dict | None]) -> str:
    counts = {"+": 0, "-": 0, "=": 0}
    for entry in entries:
        if entry and entry.get("delta") is not None and entry.get("n_paired") != 0:
            counts["=" if _mark(entry) == "?" else _mark(entry)] += 1
    return f"{counts['+']}/{counts['-']}/{counts['=']}"


def _table(suites: dict, metrics: Sequence[tuple[str, str]], *, ci: bool) -> list[str]:
    columns = [label for _, label in metrics]
    rows = [(name, [_cell(suite["headline"].get(metric), ci=ci) for metric, _ in metrics])
            for name, suite in suites.items()]
    rows.append((TALLY, [_tally([suite["headline"].get(metric) for suite in suites.values()]) for metric, _ in metrics]))
    first = max(len("Suite"), *(len(name) for name, _ in rows))
    widths = [max(len(column), *(len(cells[i]) for _, cells in rows)) for i, column in enumerate(columns)]

    def line(label, cells):
        return _SEPARATOR.join([label.ljust(first), *(cell.ljust(width) for cell, width in zip(cells, widths))]).rstrip()

    return [line("Suite", columns), *(line(name, cells) for name, cells in rows)]


def render_headline(comparison: dict, *, title: Sequence[str] = ()) -> str:
    """The headline of a comparison summary (see the module docstring), after the lines of ``title``."""
    lines = [*title, *LEGEND]
    if comparison.get("same_model"):
        lines.append("The candidate and the baseline are the same model: every Δ is 0.")
    if (comparison.get("code") or {}).get("same") is False:
        lines.append("Warning: different AlignAIR code for the two models, so every Δ holds the code change as well.")
    suites = comparison["suites"]
    lines += ["", *_table(suites, HEADLINE, ci=True)]
    for decoder, heading in DECODERS:
        lines += ["", heading,
                  *_table(suites, [(f"{metric}.{decoder}", label) for metric, label in BOUNDARIES], ci=False)]
    return "\n".join(lines)
