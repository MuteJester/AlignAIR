from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from ..adapters import normalize_call_set
from ...core.schema import BenchmarkCase, GENES

from .helpers import (
    _MISSING,
    _gene_name,
    _gene_family,
    _rate,
    _top_items,
    _add_example,
    _error_kind,
)


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
