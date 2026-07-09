"""Gene-call and gene-boundary scoring."""
from __future__ import annotations

from typing import Any

from ..adapters import normalize_call_set
from ...core.schema import BenchmarkCase, GeneTruth
from .primitives import as_float, cigar_has_gap, coord, interval_iou, levenshtein, pred_value


def _ranked_calls(pred: dict[str, Any], gene: str) -> tuple[str, ...]:
    calls = pred_value(
        pred,
        f"{gene}_ranked_calls",
        f"{gene}_candidate_calls",
        f"{gene}_candidates",
        f"{gene}_topk",
        f"{gene}_topk_calls",
    )
    if calls is None:
        scores = pred_value(pred, f"{gene}_scores", f"{gene}_allele_scores")
        if isinstance(scores, dict):
            pairs = []
            for name, score in scores.items():
                numeric = as_float(score)
                if numeric is not None:
                    pairs.append((str(name), numeric))
            if pairs:
                calls = [k for k, _ in sorted(pairs, key=lambda kv: kv[1], reverse=True)]
        elif isinstance(scores, (list, tuple)):
            pairs = []
            for item in scores:
                if isinstance(item, dict):
                    name = pred_value(item, "call", "allele", "name", "id")
                    score = as_float(pred_value(item, "score", "logit", "log_score", "prob"))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    name = item[0]
                    score = as_float(item[1])
                else:
                    continue
                if name is not None and score is not None:
                    pairs.append((str(name), score))
            if pairs:
                calls = [k for k, _ in sorted(pairs, key=lambda kv: kv[1], reverse=True)]
    if calls is None:
        return ()
    if isinstance(calls, str):
        calls = calls.split(",")
    return tuple(str(c).strip() for c in calls if str(c).strip())


def _score_topk(pred: dict[str, Any], truth: GeneTruth, gene: str) -> dict[str, float]:
    if not truth.calls:
        return {}
    ranked = _ranked_calls(pred, gene)
    if not ranked:
        return {}
    true_set = set(truth.calls)
    out = {}
    for k in (1, 3, 5, 10):
        out[f"top{k}_recall"] = 1.0 if true_set & set(ranked[:k]) else 0.0
    out["topk_truth_set_recall"] = len(true_set & set(ranked)) / len(true_set)
    true_genes = {c.split("*")[0] for c in true_set}
    if len(true_genes) == 1:
        out["same_gene_sibling_top1"] = 1.0 if ranked[0] in true_set else 0.0
        out["sibling_set_recall"] = out["topk_truth_set_recall"]
    return out


def _score_genotype_mask(
    pred: dict[str, Any],
    gene: str,
    pred_set: set[str],
    true_set: set[str],
    top1: str | None,
) -> dict[str, float]:
    allowed = pred_value(
        pred,
        f"{gene}_genotype",
        f"{gene}_allowed_calls",
        f"{gene}_candidate_mask",
        "genotype",
        "allowed_calls",
    )
    if isinstance(allowed, dict):
        allowed = allowed.get(gene) or allowed.get(gene.upper())
    if allowed is None:
        return {}
    if isinstance(allowed, str):
        allowed_set = {x.strip() for x in allowed.split(",") if x.strip()}
    else:
        allowed_set = {str(x).strip() for x in allowed if str(x).strip()}
    if not allowed_set:
        return {}
    out: dict[str, float] = {}
    if pred_set:
        out["outside_genotype_call_rate"] = sum(1 for c in pred_set if c not in allowed_set) / len(pred_set)
    # Restricted accuracy is fair only when the genotype contains a true allele.
    if true_set & allowed_set:
        out["genotype_restricted_call_acc"] = 1.0 if top1 in true_set else 0.0
    return out


def _score_gene_record_fields(pred: dict[str, Any], case: BenchmarkCase, gene: str) -> dict[str, float]:
    out: dict[str, float] = {}
    pred_cigar = pred_value(pred, f"{gene}_cigar", f"{gene}_alignment_cigar")
    truth_cigar = case.record.get(f"{gene}_cigar")
    if pred_cigar is not None and truth_cigar:
        pred_text = str(pred_cigar)
        truth_text = str(truth_cigar)
        out["cigar_exact"] = 1.0 if pred_text == truth_text else 0.0
        out["cigar_edit_distance"] = float(levenshtein(pred_text, truth_text))
        pred_gap = cigar_has_gap(pred_text)
        truth_gap = cigar_has_gap(truth_text)
        if pred_gap is not None and truth_gap is not None:
            out["gap_event_f1"] = 1.0 if pred_gap == truth_gap else 0.0

    for side in ("5", "3"):
        pred_trim = as_float(pred_value(pred, f"{gene}_trim_{side}", f"{gene}_{side}_trim"))
        truth_trim = as_float(case.record.get(f"{gene}_trim_{side}"))
        if pred_trim is not None and truth_trim is not None:
            out[f"trim_{side}_mae"] = abs(pred_trim - truth_trim)

    pred_identity = as_float(pred_value(pred, f"{gene}_identity", f"{gene}_identity_score"))
    truth_identity = as_float(case.record.get(f"{gene}_identity"))
    if pred_identity is not None and truth_identity is not None:
        out["identity_mae"] = abs(pred_identity - truth_identity)

    return out


def score_gene(
    pred: dict[str, Any],
    truth: GeneTruth,
    gene: str,
    *,
    case: BenchmarkCase,
    alt_truth: GeneTruth | None = None,
    include_expensive_record_fields: bool = True,
) -> dict[str, float]:
    if not truth.calls:
        return {}
    true_set = set(truth.calls)
    true_genes = {c.split("*")[0] for c in true_set}
    pred_calls = normalize_call_set(pred, gene)
    pred_set = set(pred_calls)
    top1 = pred_calls[0] if pred_calls else None
    intersect = true_set & pred_set
    precision = len(intersect) / len(pred_set) if pred_set else 0.0
    recall = len(intersect) / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = {
        "found_rate": 1.0 if pred_calls else 0.0,
        "missing_call_rate": 0.0 if pred_calls else 1.0,
        "call_top1_in_set": 1.0 if top1 in true_set else 0.0,
        "gene_top1_in_set": 1.0 if (top1.split("*")[0] if top1 else None) in true_genes else 0.0,
        "call_set_precision": precision,
        "call_set_recall": recall,
        "call_set_f1": f1,
        "call_exact_set": 1.0 if pred_set == true_set else 0.0,
        "truth_set_size": float(len(true_set)),
        "pred_set_size": float(len(pred_set)),
        "set_size_mae": float(abs(len(pred_set) - len(true_set))),
        "overcall_rate": 1.0 if len(pred_set) > len(true_set) else 0.0,
        "undercall_rate": 1.0 if len(pred_set) < len(true_set) else 0.0,
    }
    level = pred.get(f"{gene}_call_level")
    if level is not None:
        resolved = pred.get(f"{gene}_resolved_call")
        true_families = {c.split("-")[0] for c in true_set}
        if level == "none" or resolved is None:
            out["graceful_abstain"] = 1.0
            out["graceful_useful"] = 0.0
            out["graceful_hard_error"] = 0.0
        else:
            correct = (
                (level == "allele" and resolved in true_set)
                or (level == "gene" and resolved in true_genes)
                or (level == "family" and resolved in true_families)
            )
            out["graceful_abstain"] = 0.0
            out["graceful_useful"] = 1.0 if correct else 0.0
            out["graceful_hard_error"] = 0.0 if correct else 1.0
        out["graceful_non_error"] = 1.0 - out["graceful_hard_error"]

    coord_truth = {
        "ss": truth.sequence_start,
        "se": truth.sequence_end,
        "gs": truth.germline_start,
        "ge": truth.germline_end,
    }
    parsed = 0
    expected = 0
    off_by_one = 0
    current_err = 0.0
    alt_err = 0.0
    comparable_alt = 0
    for suffix, t in coord_truth.items():
        if t is None:
            continue
        expected += 1
        p = coord(pred, gene, suffix)
        if p is None:
            continue
        parsed += 1
        err = abs(p - float(t))
        current_err += err
        out[f"{suffix}_mae"] = err
        out[f"{suffix}_exact"] = 1.0 if err == 0 else 0.0
        out[f"{suffix}_within1"] = 1.0 if err <= 1.0 else 0.0
        out[f"{suffix}_within3"] = 1.0 if err <= 3.0 else 0.0
        out[f"{suffix}_within10"] = 1.0 if err <= 10.0 else 0.0
        off_by_one += int(err == 1.0)
        if alt_truth is not None:
            alt_t = {
                "ss": alt_truth.sequence_start,
                "se": alt_truth.sequence_end,
                "gs": alt_truth.germline_start,
                "ge": alt_truth.germline_end,
            }[suffix]
            if alt_t is not None:
                comparable_alt += 1
                alt_err += abs(p - float(alt_t))
    if expected:
        out["coordinate_parse_rate"] = parsed / expected
        out["missing_coordinate_rate"] = 1.0 - (parsed / expected)
    if parsed:
        out["off_by_one_rate"] = off_by_one / parsed
    if comparable_alt and current_err > 0 and alt_err == 0:
        out["coordinate_frame_error_rate"] = 1.0
    elif comparable_alt:
        out["coordinate_frame_error_rate"] = 0.0

    ps, pe = coord(pred, gene, "ss"), coord(pred, gene, "se")
    iou = interval_iou(ps, pe, truth.sequence_start, truth.sequence_end)
    if iou is not None:
        out["seq_span_iou"] = iou
    if ps is not None and pe is not None and truth.sequence_start is not None and truth.sequence_end is not None:
        out["segment_length_mae"] = abs((pe - ps) - (truth.sequence_end - truth.sequence_start))
        out["negative_span_rate"] = 1.0 if pe < ps else 0.0

    out.update(_score_topk(pred, truth, gene))
    out.update(_score_genotype_mask(pred, gene, pred_set, true_set, top1))
    if include_expensive_record_fields:
        out.update(_score_gene_record_fields(pred, case, gene))
    return out
