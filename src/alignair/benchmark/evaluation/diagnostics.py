"""Case-level diagnostic tables for benchmark reports."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from .adapters import normalize_call_set
from ..core.schema import BenchmarkCase, GENES, ORIENTATION_NAMES

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


class AlleleCallingDiagnosticsAccumulator:
    """Streaming accumulator for allele/gene/family confusion tables."""

    def __init__(
        self,
        *,
        frame: str = "canonical",
        top_n: int = 20,
        examples_per_row: int = 5,
    ) -> None:
        self.frame = frame
        self.top_n = top_n
        self.examples_per_row = examples_per_row
        self._genes = {gene: self._empty_gene() for gene in GENES}

    @staticmethod
    def _empty_gene() -> dict[str, Any]:
        return {
            "summary": Counter(),
            "alleles": defaultdict(Counter),
            "genes": defaultdict(Counter),
            "families": defaultdict(Counter),
            "allele_top_calls": defaultdict(Counter),
            "gene_top_genes": defaultdict(Counter),
            "gene_top_calls": defaultdict(Counter),
            "family_top_families": defaultdict(Counter),
            "family_top_genes": defaultdict(Counter),
            "family_top_calls": defaultdict(Counter),
            "allele_confusions": Counter(),
            "gene_confusions": Counter(),
            "family_confusions": Counter(),
            "allele_examples": defaultdict(list),
            "gene_examples": defaultdict(list),
            "family_examples": defaultdict(list),
        }

    def update(self, case: BenchmarkCase, prediction: dict[str, Any] | None) -> None:
        pred = prediction or {}
        for gene in GENES:
            truth = case.truth(self.frame).get(gene)
            if truth is None or not truth.calls:
                continue

            pred_calls = normalize_call_set(pred, gene)
            pred_set = set(pred_calls)
            pred_call = pred_calls[0] if pred_calls else None
            pred_gene = _gene_name(pred_call)
            pred_family = _gene_family(pred_call)
            pred_call_key = pred_call or _MISSING
            pred_gene_key = pred_gene or _MISSING
            pred_family_key = pred_family or _MISSING

            truth_calls = set(truth.calls)
            truth_genes = {g for g in (_gene_name(call) for call in truth_calls) if g}
            truth_families = {f for f in (_gene_family(call) for call in truth_calls) if f}
            error_kind = _error_kind(
                pred_call=pred_call,
                pred_gene=pred_gene,
                pred_family=pred_family,
                truth_calls=truth_calls,
                truth_genes=truth_genes,
                truth_families=truth_families,
            )

            state = self._genes[gene]
            summary = state["summary"]
            summary["n_truth_cases"] += 1
            summary["n_truth_call_slots"] += len(truth_calls)
            summary["n_predicted"] += int(pred_call is not None)
            summary["n_missing"] += int(pred_call is None)
            summary["n_top1_accepted_allele"] += int(error_kind == "accepted_allele")
            summary["n_top1_same_gene"] += int(pred_gene in truth_genes)
            summary["n_top1_same_family"] += int(pred_family in truth_families)
            summary["n_same_gene_wrong_allele"] += int(error_kind == "same_gene_wrong_allele")
            summary["n_same_family_wrong_gene"] += int(error_kind == "same_family_wrong_gene")
            summary["n_wrong_family"] += int(error_kind == "wrong_family")
            summary["n_exact_set"] += int(pred_set == truth_calls)
            summary["n_pred_set_intersects_truth"] += int(bool(pred_set & truth_calls))
            summary["n_overcalled"] += int(bool(pred_set - truth_calls))
            summary["n_undercalled"] += int(bool(truth_calls - pred_set))
            summary["pred_set_size_sum"] += len(pred_set)
            summary["truth_set_size_sum"] += len(truth_calls)

            for allele in truth_calls:
                allele_gene = _gene_name(allele)
                allele_family = _gene_family(allele)
                row = state["alleles"][allele]
                row["n_truth_cases"] += 1
                row["n_ambiguous_truth_cases"] += int(len(truth_calls) > 1)
                row["n_singleton_truth_cases"] += int(len(truth_calls) == 1)
                row["n_missing"] += int(pred_call is None)
                row["n_top1_accepted_allele"] += int(error_kind == "accepted_allele")
                row["n_pred_set_contains_allele"] += int(allele in pred_set)
                row["n_singleton_top1_exact"] += int(len(truth_calls) == 1 and pred_call == allele)
                row["n_same_gene_wrong_allele"] += int(error_kind == "same_gene_wrong_allele")
                row["n_same_family_wrong_gene"] += int(error_kind == "same_family_wrong_gene")
                row["n_wrong_family"] += int(error_kind == "wrong_family")
                row["pred_set_size_sum"] += len(pred_set)
                row["truth_set_size_sum"] += len(truth_calls)
                state["allele_top_calls"][allele][pred_call_key] += 1
                if error_kind != "accepted_allele":
                    confusion_key = (allele, pred_call_key)
                    state["allele_confusions"][confusion_key] += 1
                    _add_example(
                        state["allele_examples"],
                        confusion_key,
                        case.case_id,
                        self.examples_per_row,
                    )
                row["gene"] = allele_gene or ""
                row["family"] = allele_family or ""

            for truth_gene in truth_genes:
                row = state["genes"][truth_gene]
                row["n_truth_cases"] += 1
                row["n_missing"] += int(pred_call is None)
                row["n_top1_same_gene"] += int(pred_gene in truth_genes)
                row["n_top1_accepted_allele"] += int(error_kind == "accepted_allele")
                row["n_same_gene_wrong_allele"] += int(error_kind == "same_gene_wrong_allele")
                row["n_same_family_wrong_gene"] += int(error_kind == "same_family_wrong_gene")
                row["n_wrong_family"] += int(error_kind == "wrong_family")
                state["gene_top_genes"][truth_gene][pred_gene_key] += 1
                state["gene_top_calls"][truth_gene][pred_call_key] += 1
                if pred_gene not in truth_genes:
                    confusion_key = (truth_gene, pred_gene_key)
                    state["gene_confusions"][confusion_key] += 1
                    _add_example(
                        state["gene_examples"],
                        confusion_key,
                        case.case_id,
                        self.examples_per_row,
                    )
                row["family"] = _gene_family(truth_gene) or ""

            for truth_family in truth_families:
                row = state["families"][truth_family]
                row["n_truth_cases"] += 1
                row["n_missing"] += int(pred_call is None)
                row["n_top1_same_family"] += int(pred_family in truth_families)
                row["n_top1_same_gene"] += int(pred_gene in truth_genes)
                row["n_top1_accepted_allele"] += int(error_kind == "accepted_allele")
                row["n_same_gene_wrong_allele"] += int(error_kind == "same_gene_wrong_allele")
                row["n_same_family_wrong_gene"] += int(error_kind == "same_family_wrong_gene")
                row["n_wrong_family"] += int(error_kind == "wrong_family")
                state["family_top_families"][truth_family][pred_family_key] += 1
                state["family_top_genes"][truth_family][pred_gene_key] += 1
                state["family_top_calls"][truth_family][pred_call_key] += 1
                if pred_family not in truth_families:
                    confusion_key = (truth_family, pred_family_key)
                    state["family_confusions"][confusion_key] += 1
                    _add_example(
                        state["family_examples"],
                        confusion_key,
                        case.case_id,
                        self.examples_per_row,
                    )

    def _summary_row(self, summary: Counter) -> dict[str, Any]:
        n = summary["n_truth_cases"]
        return {
            "n_truth_cases": int(n),
            "n_truth_call_slots": int(summary["n_truth_call_slots"]),
            "n_predicted": int(summary["n_predicted"]),
            "n_missing": int(summary["n_missing"]),
            "top1_accepted_allele_rate": _rate(summary["n_top1_accepted_allele"], n),
            "top1_same_gene_rate": _rate(summary["n_top1_same_gene"], n),
            "top1_same_family_rate": _rate(summary["n_top1_same_family"], n),
            "same_gene_wrong_allele_rate": _rate(summary["n_same_gene_wrong_allele"], n),
            "same_family_wrong_gene_rate": _rate(summary["n_same_family_wrong_gene"], n),
            "wrong_family_rate": _rate(summary["n_wrong_family"], n),
            "missing_prediction_rate": _rate(summary["n_missing"], n),
            "exact_set_rate": _rate(summary["n_exact_set"], n),
            "set_intersection_rate": _rate(summary["n_pred_set_intersects_truth"], n),
            "overcall_rate": _rate(summary["n_overcalled"], n),
            "undercall_rate": _rate(summary["n_undercalled"], n),
            "mean_pred_set_size": _rate(summary["pred_set_size_sum"], n),
            "mean_truth_set_size": _rate(summary["truth_set_size_sum"], n),
        }

    def _allele_rows(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for allele, counts in state["alleles"].items():
            n = counts["n_truth_cases"]
            singleton_n = counts["n_singleton_truth_cases"]
            rows.append(
                {
                    "allele": allele,
                    "gene": counts["gene"],
                    "family": counts["family"],
                    "n_truth_cases": int(n),
                    "n_ambiguous_truth_cases": int(counts["n_ambiguous_truth_cases"]),
                    "n_singleton_truth_cases": int(singleton_n),
                    "top1_accepted_allele_rate": _rate(counts["n_top1_accepted_allele"], n),
                    "singleton_top1_exact_rate": _rate(counts["n_singleton_top1_exact"], singleton_n),
                    "pred_set_contains_allele_rate": _rate(counts["n_pred_set_contains_allele"], n),
                    "same_gene_wrong_allele_rate": _rate(counts["n_same_gene_wrong_allele"], n),
                    "same_family_wrong_gene_rate": _rate(counts["n_same_family_wrong_gene"], n),
                    "wrong_family_rate": _rate(counts["n_wrong_family"], n),
                    "missing_prediction_rate": _rate(counts["n_missing"], n),
                    "mean_pred_set_size": _rate(counts["pred_set_size_sum"], n),
                    "mean_truth_set_size": _rate(counts["truth_set_size_sum"], n),
                    "top_predicted_calls": _top_items(
                        state["allele_top_calls"][allele],
                        top_n=self.top_n,
                        key_name="call",
                    ),
                }
            )
        return sorted(
            rows,
            key=lambda row: (
                row["top1_accepted_allele_rate"] is None,
                row["top1_accepted_allele_rate"] if row["top1_accepted_allele_rate"] is not None else 1.0,
                -row["n_truth_cases"],
                row["allele"],
            ),
        )

    def _gene_rows(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for truth_gene, counts in state["genes"].items():
            n = counts["n_truth_cases"]
            rows.append(
                {
                    "gene": truth_gene,
                    "family": counts["family"],
                    "n_truth_cases": int(n),
                    "top1_same_gene_rate": _rate(counts["n_top1_same_gene"], n),
                    "top1_accepted_allele_rate": _rate(counts["n_top1_accepted_allele"], n),
                    "same_gene_wrong_allele_rate": _rate(counts["n_same_gene_wrong_allele"], n),
                    "same_family_wrong_gene_rate": _rate(counts["n_same_family_wrong_gene"], n),
                    "wrong_family_rate": _rate(counts["n_wrong_family"], n),
                    "missing_prediction_rate": _rate(counts["n_missing"], n),
                    "top_predicted_genes": _top_items(
                        state["gene_top_genes"][truth_gene],
                        top_n=self.top_n,
                        key_name="gene",
                    ),
                    "top_predicted_calls": _top_items(
                        state["gene_top_calls"][truth_gene],
                        top_n=self.top_n,
                        key_name="call",
                    ),
                }
            )
        return sorted(
            rows,
            key=lambda row: (
                row["top1_same_gene_rate"] is None,
                row["top1_same_gene_rate"] if row["top1_same_gene_rate"] is not None else 1.0,
                -row["n_truth_cases"],
                row["gene"],
            ),
        )

    def _family_rows(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for family, counts in state["families"].items():
            n = counts["n_truth_cases"]
            rows.append(
                {
                    "family": family,
                    "n_truth_cases": int(n),
                    "top1_same_family_rate": _rate(counts["n_top1_same_family"], n),
                    "top1_same_gene_rate": _rate(counts["n_top1_same_gene"], n),
                    "top1_accepted_allele_rate": _rate(counts["n_top1_accepted_allele"], n),
                    "same_gene_wrong_allele_rate": _rate(counts["n_same_gene_wrong_allele"], n),
                    "same_family_wrong_gene_rate": _rate(counts["n_same_family_wrong_gene"], n),
                    "wrong_family_rate": _rate(counts["n_wrong_family"], n),
                    "missing_prediction_rate": _rate(counts["n_missing"], n),
                    "top_predicted_families": _top_items(
                        state["family_top_families"][family],
                        top_n=self.top_n,
                        key_name="family",
                    ),
                    "top_predicted_genes": _top_items(
                        state["family_top_genes"][family],
                        top_n=self.top_n,
                        key_name="gene",
                    ),
                    "top_predicted_calls": _top_items(
                        state["family_top_calls"][family],
                        top_n=self.top_n,
                        key_name="call",
                    ),
                }
            )
        return sorted(
            rows,
            key=lambda row: (
                row["top1_same_family_rate"] is None,
                row["top1_same_family_rate"] if row["top1_same_family_rate"] is not None else 1.0,
                -row["n_truth_cases"],
                row["family"],
            ),
        )

    def _allele_confusions(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for (truth_allele, pred_call), count in state["allele_confusions"].items():
            truth_gene = _gene_name(truth_allele)
            truth_family = _gene_family(truth_allele)
            pred_call_value = None if pred_call == _MISSING else pred_call
            pred_gene = _gene_name(pred_call_value)
            pred_family = _gene_family(pred_call_value)
            rows.append(
                {
                    "truth_allele": truth_allele,
                    "truth_gene": truth_gene,
                    "truth_family": truth_family,
                    "pred_call": pred_call_value,
                    "pred_gene": pred_gene,
                    "pred_family": pred_family,
                    "n": int(count),
                    "rate_among_truth_allele_cases": _rate(
                        count,
                        state["alleles"][truth_allele]["n_truth_cases"],
                    ),
                    "error_kind": _error_kind(
                        pred_call=pred_call_value,
                        pred_gene=pred_gene,
                        pred_family=pred_family,
                        truth_calls={truth_allele},
                        truth_genes={truth_gene} if truth_gene else set(),
                        truth_families={truth_family} if truth_family else set(),
                    ),
                    "example_case_ids": state["allele_examples"][(truth_allele, pred_call)],
                }
            )
        return sorted(rows, key=lambda row: (-row["n"], row["truth_allele"], str(row["pred_call"])))[: self.top_n]

    def _gene_confusions(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for (truth_gene, pred_gene), count in state["gene_confusions"].items():
            truth_family = _gene_family(truth_gene)
            pred_gene_value = None if pred_gene == _MISSING else pred_gene
            pred_family = _gene_family(pred_gene_value)
            rows.append(
                {
                    "truth_gene": truth_gene,
                    "truth_family": truth_family,
                    "pred_gene": pred_gene_value,
                    "pred_family": pred_family,
                    "n": int(count),
                    "rate_among_truth_gene_cases": _rate(
                        count,
                        state["genes"][truth_gene]["n_truth_cases"],
                    ),
                    "error_kind": (
                        "missing_prediction"
                        if pred_gene_value is None
                        else "same_family_wrong_gene"
                        if pred_family == truth_family
                        else "wrong_family"
                    ),
                    "example_case_ids": state["gene_examples"][(truth_gene, pred_gene)],
                }
            )
        return sorted(rows, key=lambda row: (-row["n"], row["truth_gene"], str(row["pred_gene"])))[: self.top_n]

    def _family_confusions(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for (truth_family, pred_family), count in state["family_confusions"].items():
            pred_family_value = None if pred_family == _MISSING else pred_family
            rows.append(
                {
                    "truth_family": truth_family,
                    "pred_family": pred_family_value,
                    "n": int(count),
                    "rate_among_truth_family_cases": _rate(
                        count,
                        state["families"][truth_family]["n_truth_cases"],
                    ),
                    "error_kind": "missing_prediction" if pred_family_value is None else "wrong_family",
                    "example_case_ids": state["family_examples"][(truth_family, pred_family)],
                }
            )
        return sorted(rows, key=lambda row: (-row["n"], row["truth_family"], str(row["pred_family"])))[: self.top_n]

    def to_dict(self) -> dict[str, Any]:
        genes = {}
        for gene, state in self._genes.items():
            summary = self._summary_row(state["summary"])
            per_allele = self._allele_rows(state)
            per_gene = self._gene_rows(state)
            per_family = self._family_rows(state)
            summary.update(
                {
                    "n_truth_alleles": len(per_allele),
                    "n_truth_genes": len(per_gene),
                    "n_truth_families": len(per_family),
                    "n_allele_confusion_pairs": len(state["allele_confusions"]),
                    "n_gene_confusion_pairs": len(state["gene_confusions"]),
                    "n_family_confusion_pairs": len(state["family_confusions"]),
                    "per_allele_min_top1_accepted_allele_rate": (
                        min(row["top1_accepted_allele_rate"] for row in per_allele)
                        if per_allele
                        else None
                    ),
                    "per_allele_min_singleton_top1_exact_rate": (
                        min(
                            row["singleton_top1_exact_rate"]
                            for row in per_allele
                            if row["singleton_top1_exact_rate"] is not None
                        )
                        if any(row["singleton_top1_exact_rate"] is not None for row in per_allele)
                        else None
                    ),
                    "per_gene_min_top1_same_gene_rate": (
                        min(row["top1_same_gene_rate"] for row in per_gene)
                        if per_gene
                        else None
                    ),
                    "per_family_min_top1_same_family_rate": (
                        min(row["top1_same_family_rate"] for row in per_family)
                        if per_family
                        else None
                    ),
                }
            )
            genes[gene] = {
                "summary": summary,
                "per_allele": per_allele,
                "per_gene": per_gene,
                "per_gene_family": per_family,
                "allele_confusions": self._allele_confusions(state),
                "gene_confusions": self._gene_confusions(state),
                "family_confusions": self._family_confusions(state),
            }
        return {
            "truth_source": "GenAIRR benchmark cases",
            "frame": self.frame,
            "top_n": self.top_n,
            "examples_per_row": self.examples_per_row,
            "genes": genes,
        }


def build_allele_calling_diagnostics(
    cases: list[BenchmarkCase],
    predictions: list[dict[str, Any] | None],
    *,
    frame: str = "canonical",
    top_n: int = 20,
    examples_per_row: int = 5,
) -> dict[str, Any]:
    """Build allele/gene/family confusion and error tables for a prediction set."""

    if len(cases) != len(predictions):
        raise ValueError(f"case/prediction length mismatch: {len(cases)} != {len(predictions)}")
    accumulator = AlleleCallingDiagnosticsAccumulator(
        frame=frame,
        top_n=top_n,
        examples_per_row=examples_per_row,
    )
    for case, prediction in zip(cases, predictions):
        accumulator.update(case, prediction)
    return accumulator.to_dict()


def _pred_coord(pred: dict[str, Any], gene: str, suffix: str) -> float | None:
    keys = {
        "ss": (f"{gene}_sequence_start", f"{gene}_start"),
        "se": (f"{gene}_sequence_end", f"{gene}_end"),
        "gs": (f"{gene}_germline_start",),
        "ge": (f"{gene}_germline_end",),
    }[suffix]
    for key in keys:
        if key in pred:
            value = _as_float(pred.get(key))
            if value is not None:
                return value
    return None


def _truth_coords(truth) -> dict[str, float | None]:
    return {
        "ss": _as_float(truth.sequence_start),
        "se": _as_float(truth.sequence_end),
        "gs": _as_float(truth.germline_start),
        "ge": _as_float(truth.germline_end),
    }


def _span_exact(pred_start: float | None, pred_end: float | None, truth_start: float | None, truth_end: float | None) -> bool:
    return pred_start == truth_start and pred_end == truth_end


def _span_complete(start: float | None, end: float | None) -> bool:
    return start is not None and end is not None


def _boundary_contexts(case: BenchmarkCase, gene: str, truth) -> tuple[str, ...]:
    contexts = [
        f"stratum:{case.stratum}",
        f"orientation:{ORIENTATION_NAMES.get(case.orientation_id, case.orientation_id)}",
    ]
    length = len(case.sequence)
    if length <= 60:
        contexts.append("length:<=60")
    elif length <= 90:
        contexts.append("length:61-90")
    elif length <= 130:
        contexts.append("length:91-130")
    elif length <= 250:
        contexts.append("length:131-250")
    else:
        contexts.append("length:>250")
    segment_length = (
        truth.sequence_end - truth.sequence_start
        if truth.sequence_start is not None and truth.sequence_end is not None
        else None
    )
    short_thresholds = {"v": 80, "d": 8, "j": 20}
    if segment_length is not None and segment_length < short_thresholds[gene]:
        contexts.append(f"segment_visibility:short_{gene}")
    else:
        contexts.append(f"segment_visibility:{gene}_visible")
    if case.tags.get("crop_to") is not None or "fragment" in set(case.tags.get("stratum_tags", ())):
        contexts.append("fragment:yes")
    else:
        contexts.append("fragment:no")
    return tuple(contexts)


def _fragment_limited(case: BenchmarkCase, gene: str, truth) -> bool:
    if case.tags.get("crop_to") is not None or "fragment" in set(case.tags.get("stratum_tags", ())):
        return True
    if truth.sequence_start is None or truth.sequence_end is None:
        return False
    segment_length = truth.sequence_end - truth.sequence_start
    return segment_length < {"v": 80, "d": 8, "j": 20}[gene]


class BoundaryDiagnosticsAccumulator:
    """Streaming accumulator for V/D/J coordinate and span failure tables."""

    FAILURE_TYPES = (
        "missing_coordinates",
        "missing_germline_coordinates",
        "start_only_error",
        "end_only_error",
        "off_by_one",
        "germline_off_by_one",
        "systematic_plus_one_shift",
        "systematic_minus_one_shift",
        "correct_length_shifted_span",
        "wrong_length",
        "canonical_presented_frame_confusion",
        "negative_span",
        "wrong_germline_span",
        "correct_query_span_wrong_germline_span",
        "correct_allele_wrong_trim",
        "fragment_limited_boundary",
    )

    def __init__(
        self,
        *,
        frame: str = "canonical",
        top_n: int = 20,
        examples_per_row: int = 5,
    ) -> None:
        self.frame = frame
        self.top_n = top_n
        self.examples_per_row = examples_per_row
        self._genes = {gene: self._empty_gene() for gene in GENES}
        self._global = {
            "summary": Counter(),
            "failure_types": Counter(),
            "examples": defaultdict(list),
        }

    @staticmethod
    def _empty_gene() -> dict[str, Any]:
        return {
            "summary": Counter(),
            "failure_types": Counter(),
            "examples": defaultdict(list),
            "contexts": defaultdict(Counter),
            "context_examples": defaultdict(list),
        }

    def _record_failure(self, state: dict[str, Any], failure_type: str, case_id: str) -> None:
        state["failure_types"][failure_type] += 1
        _add_example(state["examples"], failure_type, case_id, self.examples_per_row)

    def _record_contexts(
        self,
        state: dict[str, Any],
        contexts: tuple[str, ...],
        *,
        exact_query: bool,
        exact_all: bool,
        failure_types: set[str],
        case_id: str,
    ) -> None:
        for context in contexts:
            row = state["contexts"][context]
            row["n_truth_segments"] += 1
            row["n_exact_query_span"] += int(exact_query)
            row["n_exact_all_coordinates"] += int(exact_all)
            for failure_type in failure_types:
                row[failure_type] += 1
                _add_example(
                    state["context_examples"],
                    (context, failure_type),
                    case_id,
                    self.examples_per_row,
                )

    def _failure_types(
        self,
        case: BenchmarkCase,
        gene: str,
        pred: dict[str, Any],
    ) -> tuple[set[str], dict[str, Any]]:
        truth = case.truth(self.frame).get(gene)
        alt_frame = "presented" if self.frame == "canonical" else "canonical"
        alt_truth = case.truth(alt_frame).get(gene)
        tc = _truth_coords(truth)
        ac = _truth_coords(alt_truth) if alt_truth is not None else {}
        pc = {suffix: _pred_coord(pred, gene, suffix) for suffix in ("ss", "se", "gs", "ge")}

        query_complete = _span_complete(pc["ss"], pc["se"])
        germline_truth_complete = _span_complete(tc["gs"], tc["ge"])
        germline_complete = _span_complete(pc["gs"], pc["ge"])
        exact_query = query_complete and _span_exact(pc["ss"], pc["se"], tc["ss"], tc["se"])
        exact_germline = germline_complete and _span_exact(pc["gs"], pc["ge"], tc["gs"], tc["ge"])
        exact_all = bool(exact_query and (not germline_truth_complete or exact_germline))
        failures: set[str] = set()

        if not query_complete:
            failures.add("missing_coordinates")
        else:
            if pc["se"] < pc["ss"]:
                failures.add("negative_span")
            if not exact_query:
                start_err = pc["ss"] - tc["ss"]
                end_err = pc["se"] - tc["se"]
                if pc["ss"] != tc["ss"] and pc["se"] == tc["se"]:
                    failures.add("start_only_error")
                if pc["ss"] == tc["ss"] and pc["se"] != tc["se"]:
                    failures.add("end_only_error")
                if abs(start_err) == 1 or abs(end_err) == 1:
                    failures.add("off_by_one")
                if start_err == 1 and end_err == 1:
                    failures.add("systematic_plus_one_shift")
                if start_err == -1 and end_err == -1:
                    failures.add("systematic_minus_one_shift")
                pred_len = pc["se"] - pc["ss"]
                truth_len = tc["se"] - tc["ss"]
                if pred_len == truth_len:
                    failures.add("correct_length_shifted_span")
                else:
                    failures.add("wrong_length")
                if alt_truth is not None and _span_exact(pc["ss"], pc["se"], ac.get("ss"), ac.get("se")):
                    failures.add("canonical_presented_frame_confusion")

        if germline_truth_complete:
            if not germline_complete:
                failures.add("missing_germline_coordinates")
            else:
                if pc["ge"] < pc["gs"]:
                    failures.add("negative_span")
                if not exact_germline:
                    failures.add("wrong_germline_span")
                    if abs(pc["gs"] - tc["gs"]) == 1 or abs(pc["ge"] - tc["ge"]) == 1:
                        failures.add("germline_off_by_one")
                    if exact_query:
                        failures.add("correct_query_span_wrong_germline_span")
                        top_call = normalize_call_set(pred, gene)
                        if top_call and top_call[0] in set(truth.calls):
                            failures.add("correct_allele_wrong_trim")

        query_failure = bool(failures & {
            "missing_coordinates",
            "start_only_error",
            "end_only_error",
            "off_by_one",
            "systematic_plus_one_shift",
            "systematic_minus_one_shift",
            "correct_length_shifted_span",
            "wrong_length",
            "canonical_presented_frame_confusion",
            "negative_span",
        })
        if query_failure and _fragment_limited(case, gene, truth):
            failures.add("fragment_limited_boundary")

        return failures, {
            "pred": pc,
            "truth": tc,
            "query_complete": query_complete,
            "germline_complete": germline_complete,
            "germline_truth_complete": germline_truth_complete,
            "exact_query": bool(exact_query),
            "exact_germline": bool(exact_germline),
            "exact_all": bool(exact_all),
        }

    def update(self, case: BenchmarkCase, prediction: dict[str, Any] | None) -> None:
        pred = prediction or {}
        order_spans = []
        order_missing = False
        order_negative = False

        for gene in GENES:
            truth = case.truth(self.frame).get(gene)
            if truth is None or not truth.present:
                continue
            state = self._genes[gene]
            summary = state["summary"]
            failures, info = self._failure_types(case, gene, pred)
            pc = info["pred"]
            tc = info["truth"]

            summary["n_truth_segments"] += 1
            summary["n_query_coordinates_complete"] += int(info["query_complete"])
            summary["n_germline_coordinates_complete"] += int(info["germline_complete"])
            summary["n_exact_query_span"] += int(info["exact_query"])
            summary["n_exact_germline_span"] += int(info["exact_germline"])
            summary["n_exact_all_coordinates"] += int(info["exact_all"])
            if info["query_complete"]:
                summary["sequence_start_abs_error_sum"] += abs(pc["ss"] - tc["ss"])
                summary["sequence_end_abs_error_sum"] += abs(pc["se"] - tc["se"])
                summary["segment_length_abs_error_sum"] += abs((pc["se"] - pc["ss"]) - (tc["se"] - tc["ss"]))
            if info["germline_complete"] and info["germline_truth_complete"]:
                summary["germline_start_abs_error_sum"] += abs(pc["gs"] - tc["gs"])
                summary["germline_end_abs_error_sum"] += abs(pc["ge"] - tc["ge"])

            for failure_type in failures:
                self._record_failure(state, failure_type, case.case_id)

            self._record_contexts(
                state,
                _boundary_contexts(case, gene, truth),
                exact_query=info["exact_query"],
                exact_all=info["exact_all"],
                failure_types=failures,
                case_id=case.case_id,
            )

            if info["query_complete"]:
                order_spans.append((gene, pc["ss"], pc["se"]))
                order_negative = order_negative or pc["se"] < pc["ss"]
            else:
                order_missing = True

        self._update_order(case, order_spans, missing=order_missing, negative=order_negative)

    def _update_order(
        self,
        case: BenchmarkCase,
        spans: list[tuple[str, float, float]],
        *,
        missing: bool,
        negative: bool,
    ) -> None:
        present_genes = [g for g in GENES if case.truth(self.frame).get(g) and case.truth(self.frame)[g].present]
        if len(present_genes) < 2:
            return
        summary = self._global["summary"]
        summary["n_cases_with_multiple_segments"] += 1
        if missing:
            self._global["failure_types"]["vdj_order_missing_coordinates"] += 1
            _add_example(
                self._global["examples"],
                "vdj_order_missing_coordinates",
                case.case_id,
                self.examples_per_row,
            )
            return
        if negative:
            self._global["failure_types"]["vdj_order_or_overlap_error"] += 1
            _add_example(
                self._global["examples"],
                "vdj_order_or_overlap_error",
                case.case_id,
                self.examples_per_row,
            )
            return
        spans_by_gene = {gene: (start, end) for gene, start, end in spans}
        ordered_spans = [(gene, *spans_by_gene[gene]) for gene in present_genes if gene in spans_by_gene]
        ordered = True
        for (_, _, prev_end), (_, next_start, _) in zip(ordered_spans, ordered_spans[1:]):
            if next_start < prev_end:
                ordered = False
                break
        summary["n_vdj_order_valid"] += int(ordered)
        if not ordered:
            self._global["failure_types"]["vdj_order_or_overlap_error"] += 1
            _add_example(
                self._global["examples"],
                "vdj_order_or_overlap_error",
                case.case_id,
                self.examples_per_row,
            )

    def _summary_row(self, summary: Counter) -> dict[str, Any]:
        n = summary["n_truth_segments"]
        query_n = summary["n_query_coordinates_complete"]
        germ_n = summary["n_germline_coordinates_complete"]
        row = {
            "n_truth_segments": int(n),
            "query_coordinate_parse_rate": _rate(query_n, n),
            "germline_coordinate_parse_rate": _rate(germ_n, n),
            "exact_query_span_rate": _rate(summary["n_exact_query_span"], n),
            "exact_germline_span_rate": _rate(summary["n_exact_germline_span"], n),
            "exact_all_coordinates_rate": _rate(summary["n_exact_all_coordinates"], n),
            "sequence_start_mae": _rate(summary["sequence_start_abs_error_sum"], query_n),
            "sequence_end_mae": _rate(summary["sequence_end_abs_error_sum"], query_n),
            "segment_length_mae": _rate(summary["segment_length_abs_error_sum"], query_n),
            "germline_start_mae": _rate(summary["germline_start_abs_error_sum"], germ_n),
            "germline_end_mae": _rate(summary["germline_end_abs_error_sum"], germ_n),
        }
        return row

    def _failure_rows(self, state: dict[str, Any], denominator: int) -> list[dict[str, Any]]:
        rows = []
        for failure_type in self.FAILURE_TYPES:
            count = state["failure_types"].get(failure_type, 0)
            if count == 0:
                continue
            rows.append(
                {
                    "failure_type": failure_type,
                    "n": int(count),
                    "rate": _rate(count, denominator),
                    "example_case_ids": state["examples"][failure_type],
                }
            )
        rows.sort(key=lambda row: (-row["n"], row["failure_type"]))
        return rows

    def _context_rows(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for context, counts in state["contexts"].items():
            n = counts["n_truth_segments"]
            row = {
                "context": context,
                "n_truth_segments": int(n),
                "exact_query_span_rate": _rate(counts["n_exact_query_span"], n),
                "exact_all_coordinates_rate": _rate(counts["n_exact_all_coordinates"], n),
            }
            failures = []
            for failure_type in self.FAILURE_TYPES:
                count = counts.get(failure_type, 0)
                if count:
                    failures.append(
                        {
                            "failure_type": failure_type,
                            "n": int(count),
                            "rate": _rate(count, n),
                            "example_case_ids": state["context_examples"][(context, failure_type)],
                        }
                    )
            failures.sort(key=lambda item: (-item["n"], item["failure_type"]))
            row["failure_types"] = failures[: self.top_n]
            rows.append(row)
        rows.sort(
            key=lambda row: (
                row["exact_query_span_rate"] is None,
                row["exact_query_span_rate"] if row["exact_query_span_rate"] is not None else 1.0,
                -row["n_truth_segments"],
                row["context"],
            )
        )
        return rows

    def _global_row(self) -> dict[str, Any]:
        summary = self._global["summary"]
        n = summary["n_cases_with_multiple_segments"]
        return {
            "summary": {
                "n_cases_with_multiple_segments": int(n),
                "vdj_order_valid_rate": _rate(summary["n_vdj_order_valid"], n),
            },
            "failure_types": [
                {
                    "failure_type": failure_type,
                    "n": int(count),
                    "rate": _rate(count, n),
                    "example_case_ids": self._global["examples"][failure_type],
                }
                for failure_type, count in sorted(
                    self._global["failure_types"].items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        genes = {}
        for gene, state in self._genes.items():
            summary = self._summary_row(state["summary"])
            n = summary["n_truth_segments"]
            summary.update(
                {
                    f"{failure_type}_rate": _rate(state["failure_types"].get(failure_type, 0), n)
                    for failure_type in self.FAILURE_TYPES
                }
            )
            genes[gene] = {
                "summary": summary,
                "failure_types": self._failure_rows(state, n),
                "by_context": self._context_rows(state),
            }
        return {
            "truth_source": "GenAIRR benchmark cases",
            "frame": self.frame,
            "top_n": self.top_n,
            "examples_per_row": self.examples_per_row,
            "genes": genes,
            "global": self._global_row(),
        }


def build_boundary_diagnostics(
    cases: list[BenchmarkCase],
    predictions: list[dict[str, Any] | None],
    *,
    frame: str = "canonical",
    top_n: int = 20,
    examples_per_row: int = 5,
) -> dict[str, Any]:
    """Build V/D/J coordinate failure diagnostics for a prediction set."""

    if len(cases) != len(predictions):
        raise ValueError(f"case/prediction length mismatch: {len(cases)} != {len(predictions)}")
    accumulator = BoundaryDiagnosticsAccumulator(
        frame=frame,
        top_n=top_n,
        examples_per_row=examples_per_row,
    )
    for case, prediction in zip(cases, predictions):
        accumulator.update(case, prediction)
    return accumulator.to_dict()
