from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from ..adapters import normalize_call_set
from ...core.schema import BenchmarkCase, GENES, ORIENTATION_NAMES

from .helpers import (
    _as_float,
    _rate,
    _add_example,
)


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
