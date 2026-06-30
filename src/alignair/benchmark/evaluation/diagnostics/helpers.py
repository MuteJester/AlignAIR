from __future__ import annotations

import re
from collections import Counter
from typing import Any

_FAMILY_RE = re.compile(r"^([A-Za-z]+[0-9]+)")
_MISSING = "<missing>"


def _gene_name(call: str | None) -> str | None:
    if not call:
        return None
    return str(call).split("*", 1)[0]


def _gene_family(call: str | None) -> str | None:
    gene = _gene_name(call)
    if not gene:
        return None
    match = _FAMILY_RE.match(gene)
    return match.group(1) if match else gene.split("-", 1)[0]


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _top_items(counter: Counter[str], *, top_n: int, key_name: str) -> list[dict[str, Any]]:
    rows = []
    for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:top_n]:
        rows.append({key_name: None if value == _MISSING else value, "n": int(count)})
    return rows


def _add_example(examples: dict[Any, list[str]], key: Any, case_id: str, limit: int) -> None:
    if limit <= 0:
        return
    rows = examples[key]
    if len(rows) < limit:
        rows.append(case_id)


def _error_kind(
    *,
    pred_call: str | None,
    pred_gene: str | None,
    pred_family: str | None,
    truth_calls: set[str],
    truth_genes: set[str],
    truth_families: set[str],
) -> str:
    if pred_call is None:
        return "missing_prediction"
    if pred_call in truth_calls:
        return "accepted_allele"
    if pred_gene in truth_genes:
        return "same_gene_wrong_allele"
    if pred_family in truth_families:
        return "same_family_wrong_gene"
    return "wrong_family"
