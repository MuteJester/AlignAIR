"""Machine-readable report dictionaries (strictly JSON-serializable: NaN and infinity become None)."""
from __future__ import annotations

import math

from alignair_assay.stats import to_dict

SCORE_KIND = "alignair_assay.score_report"
COMPARISON_KIND = "alignair_assay.comparison_report"
VERSION = "1"


def _clean(obj):
    obj = to_dict(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def build_score_report(tool: str, summary, slices, provenance: dict, *, excess_alleles: dict | None = None
                       ) -> dict:
    """Report for one tool: overall summaries plus per-dimension slice summaries, and the donor-level
    ``excess_alleles`` table (``metrics.genotype.excess_alleles``) when one is given."""
    report = {"kind": SCORE_KIND, "version": VERSION, "tool": tool, "overall": _clean(summary),
              "by_slice": _clean(slices), "provenance": _clean(provenance)}
    if excess_alleles is not None:
        report["excess_alleles"] = _clean(excess_alleles)
    return report


def build_comparison_report(a_name: str, b_name: str, comparison, slices_a, slices_b, provenance: dict, *,
                            informational: dict | None = None) -> dict:
    """Report for a paired comparison of tool ``a`` against tool ``b``: the tested ``comparison`` and, as
    ``informational``, the metrics reported but not compared (name -> ``{"a": summary, "b": summary}``, each tool's
    own mean with n and coverage, None when it has none)."""
    return {"kind": COMPARISON_KIND, "version": VERSION, "a": a_name, "b": b_name,
            "comparison": _clean(comparison), "informational": _clean(informational or {}),
            "by_slice": {"a": _clean(slices_a), "b": _clean(slices_b)}, "provenance": _clean(provenance)}
