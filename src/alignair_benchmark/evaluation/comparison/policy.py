"""Comparison policy templates and validation."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from ...core import GENES, metric_registry, metric_spec

MULTIPLE_COMPARISON_CORRECTIONS: tuple[str, ...] = ("none", "bonferroni", "sidak")
COMPARISON_POLICY_TEMPLATES: dict[str, dict[str, Any]] = {
    "allele_calling_core": {
        "description": "Mixed-chain V/J allele-call endpoint with output, set-call, boundary, and junction guardrails.",
        "primary_metrics": (
            "genes.v.call_top1_in_set",
            "genes.j.call_top1_in_set",
        ),
        "guardrail_metrics": (
            "genes.v.call_set_f1",
            "genes.j.call_set_f1",
            "genes.v.ss_mae",
            "genes.v.se_mae",
            "genes.j.ss_mae",
            "genes.j.se_mae",
            "global.junction_nt_exact",
            "global.required_field_presence",
            "global.parseable_airr_rate",
        ),
        "minimum_primary_advantage": 0.0,
        "maximum_guardrail_regression": 0.0,
    },
    "igh_allele_calling_core": {
        "description": "IGH V/D/J allele-call endpoint with set-call, boundary, junction, and AIRR output guardrails.",
        "primary_metrics": (
            "genes.v.call_top1_in_set",
            "genes.d.call_top1_in_set",
            "genes.j.call_top1_in_set",
        ),
        "guardrail_metrics": (
            "genes.v.call_set_f1",
            "genes.d.call_set_f1",
            "genes.j.call_set_f1",
            "genes.v.ss_mae",
            "genes.v.se_mae",
            "genes.d.ss_mae",
            "genes.d.se_mae",
            "genes.j.ss_mae",
            "genes.j.se_mae",
            "global.junction_nt_exact",
            "global.productive_acc",
            "global.required_field_presence",
            "global.parseable_airr_rate",
        ),
        "minimum_primary_advantage": 0.0,
        "maximum_guardrail_regression": 0.0,
    },
    "boundary_core": {
        "description": "V/J query-boundary endpoint with allele-call, junction, and AIRR output guardrails.",
        "primary_metrics": (
            "genes.v.ss_mae",
            "genes.v.se_mae",
            "genes.j.ss_mae",
            "genes.j.se_mae",
        ),
        "guardrail_metrics": (
            "genes.v.call_top1_in_set",
            "genes.j.call_top1_in_set",
            "global.junction_nt_exact",
            "global.required_field_presence",
            "global.parseable_airr_rate",
        ),
        "minimum_primary_advantage": 0.0,
        "maximum_guardrail_regression": 0.0,
    },
    "airr_core": {
        "description": (
            "General AIRR-style endpoint over V/J calls, junction recovery, "
            "productivity, and output completeness."
        ),
        "primary_metrics": (
            "genes.v.call_top1_in_set",
            "genes.j.call_top1_in_set",
            "global.junction_nt_exact",
            "global.productive_acc",
        ),
        "guardrail_metrics": (
            "genes.v.call_set_f1",
            "genes.j.call_set_f1",
            "genes.v.ss_mae",
            "genes.v.se_mae",
            "genes.j.ss_mae",
            "genes.j.se_mae",
            "global.required_field_presence",
            "global.parseable_airr_rate",
        ),
        "minimum_primary_advantage": 0.0,
        "maximum_guardrail_regression": 0.0,
    },
}


def metric_key_from_path(path: str) -> str:
    return path.split(".")[-1]


def comparison_policy_catalog() -> list[dict[str, Any]]:
    """Return the built-in paired-comparison endpoint policies."""

    out = []
    for name, policy in sorted(COMPARISON_POLICY_TEMPLATES.items()):
        row = {"name": name}
        for key, value in policy.items():
            row[key] = list(value) if isinstance(value, tuple) else value
        out.append(row)
    return out


def policy_template(name: str | None) -> dict[str, Any] | None:
    if name is None:
        return None
    if name not in COMPARISON_POLICY_TEMPLATES:
        choices = ", ".join(sorted(COMPARISON_POLICY_TEMPLATES))
        raise ValueError(f"unknown comparison policy {name!r}; expected one of: {choices}")
    return COMPARISON_POLICY_TEMPLATES[name]


def _metric_path_problem(path: str) -> str | None:
    parts = path.split(".")
    if len(parts) == 2 and parts[0] == "global" and parts[1]:
        return None
    if len(parts) == 3 and parts[0] == "genes" and parts[1] in GENES and parts[2]:
        return None
    return "malformed_metric_path"


def validate_comparison_policy_catalog(
    policies: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate comparison policy metric endpoints against the metric registry."""

    policy_map = policies or COMPARISON_POLICY_TEMPLATES
    registry_keys = set(metric_registry())
    rows = []
    problem_counts: dict[str, int] = defaultdict(int)
    for policy_name, policy in sorted(policy_map.items()):
        for role_key, role in (("primary_metrics", "primary"), ("guardrail_metrics", "guardrail")):
            for path in policy.get(role_key, ()):
                key = metric_key_from_path(path)
                problems = []
                path_problem = _metric_path_problem(path)
                if path_problem:
                    problems.append(path_problem)
                if key not in registry_keys:
                    problems.append("metric_key_not_in_registry")
                for problem in problems:
                    problem_counts[problem] += 1
                spec = metric_spec(key)
                rows.append(
                    {
                        "policy": policy_name,
                        "role": role,
                        "metric": path,
                        "metric_key": key,
                        "registered": key in registry_keys,
                        "higher_is_better": spec.higher_is_better,
                        "problems": tuple(problems),
                    }
                )
    invalid_rows = [row for row in rows if row["problems"]]
    return {
        "valid": not invalid_rows,
        "summary": {
            "n_policies": len(policy_map),
            "n_policy_metrics": len(rows),
            "n_invalid_policy_metrics": len(invalid_rows),
            "problem_counts": dict(sorted(problem_counts.items())),
        },
        "policy_metrics": rows,
    }
