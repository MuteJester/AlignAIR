"""Per-case metrics: one tool's prediction for one read, scored against observable truth.

``CaseScores`` maps a metric name (``"<scope>.<metric>"``, e.g. ``"v.top1_in_set"``) to a value. An absent key
means the metric does not apply to the case. An applicable metric the tool did not cover is ``nan``, except
accuracy-type metrics, which score a missing prediction as 0.0 and so are never ``nan``.
"""
from __future__ import annotations

CaseScores = dict[str, float]

# Metric-name suffixes that are reported but excluded from tool comparisons.
INFORMATIONAL: frozenset[str] = frozenset({".set_size"})

_LOWER_IS_BETTER = ("_err", ".no_call", ".spurious_call", ".germline_errors", ".mutation_count_err")


def lower_is_better(name: str) -> bool:
    """True for metrics where a smaller value is better (errors, failure rates and counts)."""
    return name.endswith(_LOWER_IS_BETTER)
