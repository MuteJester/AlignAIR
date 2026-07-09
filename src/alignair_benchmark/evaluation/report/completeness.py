from __future__ import annotations

from typing import Any


def _apply_completeness_gate(
    entries: list[dict[str, Any]],
    source_report: dict[str, Any],
) -> dict[str, Any]:
    """Fail full reports when available core criteria are unscored."""

    audit = source_report.get("criteria_audit") or {}
    audit_summary = audit.get("summary") or {}
    has_truth_audit = bool(audit_summary.get("has_case_truth_audit"))
    if not has_truth_audit:
        return {
            "applied": False,
            "grade": "pass",
            "reason": "case-level truth audit unavailable; completeness gate was not applied",
            "blocking_unscored_core_criteria": [],
            "truth_unavailable_core_criteria": [],
            "partial_unscored_core_criteria": [],
        }

    audit_by_name = {
        row.get("name"): row
        for row in audit.get("criteria", ())
        if row.get("name")
    }
    blocking = []
    truth_unavailable = []
    partial_unscored = []
    for entry in entries:
        if entry.get("importance") != "core" or entry.get("n_observed_metric_keys", 0) > 0:
            continue
        audit_row = audit_by_name.get(entry["name"], {})
        unavailable_truth = tuple(audit_row.get("unavailable_truth_fields", ()))
        row = {
            "name": entry["name"],
            "category": entry["category"],
            "status": entry.get("status"),
            "missing_metric_keys": entry.get("missing_metric_keys", ()),
            "unavailable_truth_fields": unavailable_truth,
        }
        if entry.get("status") == "available":
            if unavailable_truth:
                truth_unavailable.append(row)
            else:
                blocking.append(row)
                entry["grade"] = "fail"
                entry["grade_reasons"] = list(entry.get("grade_reasons", ())) + [
                    "available core criterion has no observed metrics despite available truth fields",
                ]
                entry["completeness_gate_failed"] = True
        elif entry.get("status") == "partial":
            partial_unscored.append(row)

    return {
        "applied": True,
        "grade": "fail" if blocking else "pass",
        "reason": (
            "available core criteria without observed metrics block a pass"
            if blocking
            else "no available core criteria were missing scoreable metrics"
        ),
        "blocking_unscored_core_criteria": blocking,
        "truth_unavailable_core_criteria": truth_unavailable,
        "partial_unscored_core_criteria": partial_unscored,
    }
