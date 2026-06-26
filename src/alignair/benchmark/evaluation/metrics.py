"""Benchmark metrics for AIRR alignment predictions."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable

from .adapters import normalize_call_set
from .performance import prediction_performance_metrics
from ..core.schema import BenchmarkCase, GENES, GeneTruth
from ...nn.heads.region import REGIONS
from ...nn.heads.state import STATE_INDEX, STATES

_CIGAR_RE = re.compile(r"([0-9]+)([A-Za-z=])")

_AIRR_REQUIRED = (
    "sequence_id",
    "sequence",
    "v_call",
    "j_call",
    "productive",
    "junction",
)
_AIRR_OPTIONAL = (
    "d_call",
    "c_call",
    "junction_aa",
    "vj_in_frame",
    "stop_codon",
    "v_cigar",
    "d_cigar",
    "j_cigar",
    "v_identity",
    "d_identity",
    "j_identity",
)


def _avg(values: list[float]) -> float:
    return float(mean(values)) if values else float("nan")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _as_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "t", "true", "yes", "y", "productive", "in_frame", "inverted"}:
        return True
    if text in {"0", "f", "false", "no", "n", "nonproductive", "out_of_frame", "forward"}:
        return False
    return None


def _pred_value(pred: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in pred and not _is_missing(pred[key]):
            return pred[key]
    return None


def _field_presence(pred: dict[str, Any], keys: tuple[str, ...]) -> float:
    if not keys:
        return float("nan")
    present = sum(1 for key in keys if key in pred and not _is_missing(pred[key]))
    return present / len(keys)


def _coord(pred: dict[str, Any], gene: str, suffix: str) -> float | None:
    keys = {
        "ss": (f"{gene}_sequence_start", f"{gene}_start"),
        "se": (f"{gene}_sequence_end", f"{gene}_end"),
        "gs": (f"{gene}_germline_start",),
        "ge": (f"{gene}_germline_end",),
    }[suffix]
    return _as_float(_pred_value(pred, *keys))


def _string_exact(pred_value: Any, truth_value: Any, *, case_sensitive: bool = False) -> float | None:
    if pred_value is None or truth_value is None:
        return None
    pred_text = str(pred_value)
    truth_text = str(truth_value)
    if not case_sensitive:
        pred_text = pred_text.upper()
        truth_text = truth_text.upper()
    return 1.0 if pred_text == truth_text else 0.0


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _cigar_has_gap(cigar: Any) -> bool | None:
    if _is_missing(cigar):
        return None
    return any(op in {"I", "D", "N"} for _, op in _CIGAR_RE.findall(str(cigar)))


def _interval_iou(
    pred_start: float | None,
    pred_end: float | None,
    truth_start: float | None,
    truth_end: float | None,
) -> float | None:
    if None in (pred_start, pred_end, truth_start, truth_end):
        return None
    inter = max(0.0, min(pred_end, truth_end) - max(pred_start, truth_start))
    union = max(pred_end, truth_end) - min(pred_start, truth_start)
    return inter / union if union > 0 else 0.0


def _oriented_interval(case: BenchmarkCase, start_key: str, end_key: str, frame: str) -> tuple[float | None, float | None]:
    start = _as_float(case.record.get(start_key))
    end = _as_float(case.record.get(end_key))
    if start is None or end is None:
        return None, None
    if frame == "presented" and case.orientation_id in (1, 3):
        length = len(case.canonical_sequence)
        return float(length) - end, float(length) - start
    return start, end


def _binary_prf(pred_labels: list[Any], truth_labels: list[int], label_id: int) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for pred, truth in zip(pred_labels, truth_labels):
        p = int(pred) == label_id
        t = int(truth) == label_id
        tp += int(p and t)
        fp += int(p and not t)
        fn += int((not p) and t)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _binary_set_prf(
    pred_labels: list[Any],
    truth_labels: list[int],
    label_ids: set[int],
) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for pred, truth in zip(pred_labels, truth_labels):
        p = int(pred) in label_ids
        t = int(truth) in label_ids
        tp += int(p and t)
        fp += int(p and not t)
        fn += int((not p) and t)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _score_labels(pred: dict[str, Any], case: BenchmarkCase, name: str, frame: str) -> dict[str, float]:
    pred_labels = pred.get(f"{name}_labels")
    if pred_labels is None:
        return {}
    truth = case.labels(name, frame)
    n = min(len(pred_labels), len(truth))
    if n == 0:
        return {}
    pred_slice = list(pred_labels[:n])
    truth_slice = list(truth[:n])
    correct = sum(1 for a, b in zip(pred_slice, truth_slice) if int(a) == int(b))
    out = {f"{name}_acc": correct / n}

    labels = range(len(REGIONS) if name == "region" else len(STATES))
    class_recalls = []
    class_f1s = []
    for label_id in labels:
        precision, recall, f1 = _binary_prf(pred_slice, truth_slice, label_id)
        if any(int(t) == label_id for t in truth_slice):
            class_recalls.append(recall)
            class_f1s.append(f1)
        label_name = REGIONS[label_id] if name == "region" else STATES[label_id]
        out[f"{name}_{label_name.lower()}_recall"] = recall
        out[f"{name}_{label_name.lower()}_f1"] = f1
        if name == "state" and label_name in {"substitution", "insertion", "deletion"}:
            out[f"{label_name}_precision"] = precision
            out[f"{label_name}_recall"] = recall
            out[f"{label_name}_f1"] = f1
    if class_recalls:
        out[f"{name}_per_class_recall"] = _avg(class_recalls)
        out[f"{name}_macro_f1"] = _avg(class_f1s)
    if name == "state":
        sub = STATE_INDEX["substitution"]
        p, r, f1 = _binary_prf(pred_slice, truth_slice, sub)
        out["shm_site_precision"] = p
        out["shm_site_recall"] = r
        out["shm_site_f1"] = f1
        out["false_shm_from_noise_rate"] = 1.0 - p
        indel_ids = {STATE_INDEX["insertion"], STATE_INDEX["deletion"]}
        p, r, f1 = _binary_set_prf(pred_slice, truth_slice, indel_ids)
        out["indel_event_precision"] = p
        out["indel_event_recall"] = r
        out["indel_event_f1"] = f1
    return out


def _score_scalar(pred: dict[str, Any], case: BenchmarkCase, name: str) -> dict[str, float]:
    if name not in pred or name not in case.scalars or pred[name] is None:
        return {}
    truth = case.scalars[name]
    p = _as_float(pred[name])
    if p is None:
        return {}
    if name == "productive":
        return {"productive_acc": 1.0 if (p >= 0.5) == (truth >= 0.5) else 0.0}
    if name == "indel_count":
        return {"indel_count_mae": abs(p - truth), "indel_length_mae": abs(p - truth)}
    return {f"{name}_mae": abs(p - truth)}


def _ranked_calls(pred: dict[str, Any], gene: str) -> tuple[str, ...]:
    calls = _pred_value(
        pred,
        f"{gene}_ranked_calls",
        f"{gene}_candidate_calls",
        f"{gene}_candidates",
        f"{gene}_topk",
        f"{gene}_topk_calls",
    )
    if calls is None:
        scores = _pred_value(pred, f"{gene}_scores", f"{gene}_allele_scores")
        if isinstance(scores, dict):
            pairs = []
            for name, score in scores.items():
                numeric = _as_float(score)
                if numeric is not None:
                    pairs.append((str(name), numeric))
            if pairs:
                calls = [k for k, _ in sorted(pairs, key=lambda kv: kv[1], reverse=True)]
        elif isinstance(scores, (list, tuple)):
            pairs = []
            for item in scores:
                if isinstance(item, dict):
                    name = _pred_value(item, "call", "allele", "name", "id")
                    score = _as_float(_pred_value(item, "score", "logit", "log_score", "prob"))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    name = item[0]
                    score = _as_float(item[1])
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


def _score_genotype_mask(pred: dict[str, Any], gene: str, pred_set: set[str],
                         true_set: set[str], top1: str | None) -> dict[str, float]:
    allowed = _pred_value(
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
    # restricted-accuracy is only a fair test when the genotype actually CONTAINS a true
    # allele (a realistic donor genotype does); otherwise the right answer isn't available.
    if true_set & allowed_set:
        out["genotype_restricted_call_acc"] = 1.0 if top1 in true_set else 0.0
    return out


def _score_gene_record_fields(pred: dict[str, Any], case: BenchmarkCase, gene: str) -> dict[str, float]:
    out: dict[str, float] = {}
    pred_cigar = _pred_value(pred, f"{gene}_cigar", f"{gene}_alignment_cigar")
    truth_cigar = case.record.get(f"{gene}_cigar")
    if pred_cigar is not None and truth_cigar:
        pred_text = str(pred_cigar)
        truth_text = str(truth_cigar)
        out["cigar_exact"] = 1.0 if pred_text == truth_text else 0.0
        out["cigar_edit_distance"] = float(_levenshtein(pred_text, truth_text))
        pred_gap = _cigar_has_gap(pred_text)
        truth_gap = _cigar_has_gap(truth_text)
        if pred_gap is not None and truth_gap is not None:
            out["gap_event_f1"] = 1.0 if pred_gap == truth_gap else 0.0

    for side in ("5", "3"):
        pred_trim = _as_float(_pred_value(pred, f"{gene}_trim_{side}", f"{gene}_{side}_trim"))
        truth_trim = _as_float(case.record.get(f"{gene}_trim_{side}"))
        if pred_trim is not None and truth_trim is not None:
            out[f"trim_{side}_mae"] = abs(pred_trim - truth_trim)

    pred_identity = _as_float(_pred_value(pred, f"{gene}_identity", f"{gene}_identity_score"))
    truth_identity = _as_float(case.record.get(f"{gene}_identity"))
    if pred_identity is not None and truth_identity is not None:
        out["identity_mae"] = abs(pred_identity - truth_identity)

    return out


def _score_gene(
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
    # graceful hierarchical degradation (optional {gene}_resolved_call/{gene}_call_level):
    # credit a correct coarser call or an honest abstention instead of a forced wrong allele.
    level = pred.get(f"{gene}_call_level")
    if level is not None:
        resolved = pred.get(f"{gene}_resolved_call")
        true_families = {c.split("-")[0] for c in true_set}
        if level == "none" or resolved is None:
            out["graceful_abstain"] = 1.0
            out["graceful_useful"] = 0.0
            out["graceful_hard_error"] = 0.0          # abstaining is not an error
        else:
            correct = ((level == "allele" and resolved in true_set)
                       or (level == "gene" and resolved in true_genes)
                       or (level == "family" and resolved in true_families))
            out["graceful_abstain"] = 0.0
            out["graceful_useful"] = 1.0 if correct else 0.0
            out["graceful_hard_error"] = 0.0 if correct else 1.0
        # non-error = useful OR honest abstention (the real quality target: never confidently
        # wrong). Graded higher-is-better; raw `useful` alone wrongly penalizes correct
        # abstention on information-limited reads, so it stays informational only.
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
        p = _coord(pred, gene, suffix)
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

    ps, pe = _coord(pred, gene, "ss"), _coord(pred, gene, "se")
    iou = _interval_iou(ps, pe, truth.sequence_start, truth.sequence_end)
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


def _score_airr_contract(pred: dict[str, Any], case: BenchmarkCase) -> dict[str, float]:
    required = list(_AIRR_REQUIRED)
    if case.genes.get("d") and case.genes["d"].calls:
        required.append("d_call")
    out = {
        "required_field_presence": _field_presence(pred, tuple(required)),
        "optional_field_presence": _field_presence(pred, _AIRR_OPTIONAL),
    }
    coord_keys = tuple(
        f"{gene}_{kind}_{side}"
        for gene in GENES
        for kind in ("sequence", "germline")
        for side in ("start", "end")
    )
    parseable = [
        _as_float(pred.get(key)) is not None
        for key in coord_keys
        if key in pred and not _is_missing(pred.get(key))
    ]
    out["parseable_airr_rate"] = 1.0 if parseable and all(parseable) else 0.0
    return out


def _score_segment_order(pred: dict[str, Any], case: BenchmarkCase, frame: str) -> dict[str, float]:
    truth = case.truth(frame)
    present_genes = [g for g in GENES if truth.get(g) and truth[g].present]
    if not present_genes:
        return {}
    spans = []
    missing = False
    negative = 0
    for gene in present_genes:
        start = _coord(pred, gene, "ss")
        end = _coord(pred, gene, "se")
        if start is None or end is None:
            missing = True
            continue
        negative += int(end < start)
        spans.append((gene, start, end))
    if not spans:
        return {"vdj_order_valid": 0.0, "overlap_rate": 1.0, "negative_span_rate": 1.0}
    ordered = True
    overlaps = 0
    pair_count = 0
    for (_, _, prev_end), (_, next_start, _) in zip(spans, spans[1:]):
        pair_count += 1
        if next_start < prev_end:
            overlaps += 1
            ordered = False
    if missing or negative:
        ordered = False
    return {
        "vdj_order_valid": 1.0 if ordered else 0.0,
        "overlap_rate": overlaps / pair_count if pair_count else 0.0,
        "negative_span_rate": negative / len(present_genes),
    }


def _score_junction(pred: dict[str, Any], case: BenchmarkCase, frame: str) -> dict[str, float]:
    record = case.record
    out: dict[str, float] = {}
    truth_start, truth_end = _oriented_interval(case, "junction_start", "junction_end", frame)
    pred_start = _as_float(_pred_value(pred, "junction_start", "cdr3_start"))
    pred_end = _as_float(_pred_value(pred, "junction_end", "cdr3_end"))
    if pred_start is not None and truth_start is not None:
        out["junction_start_mae"] = abs(pred_start - truth_start)
    if pred_end is not None and truth_end is not None:
        out["junction_end_mae"] = abs(pred_end - truth_end)
    iou = _interval_iou(pred_start, pred_end, truth_start, truth_end)
    if iou is not None:
        out["cdr3_overlap_iou"] = iou

    pred_junction = _pred_value(pred, "junction", "cdr3")
    exact = _string_exact(pred_junction, record.get("junction"))
    if exact is not None:
        out["junction_nt_exact"] = exact
    pred_junction_aa = _pred_value(pred, "junction_aa", "cdr3_aa")
    exact_aa = _string_exact(pred_junction_aa, record.get("junction_aa"))
    if exact_aa is not None:
        out["junction_aa_exact"] = exact_aa

    truth_len = _as_float(record.get("junction_length"))
    pred_len = _as_float(_pred_value(pred, "junction_length", "cdr3_length"))
    if pred_len is None and pred_junction is not None:
        pred_len = float(len(str(pred_junction)))
    if pred_len is not None and truth_len is not None:
        out["junction_length_mae"] = abs(pred_len - truth_len)
    if pred_junction_aa is not None and record.get("junction_aa") is not None:
        out["junction_aa_length_mae"] = abs(len(str(pred_junction_aa)) - len(str(record.get("junction_aa"))))

    for region, metric in (("np1", "n1"), ("np2", "n2")):
        pred_seq = _pred_value(pred, region, metric)
        exact_region = _string_exact(pred_seq, record.get(region))
        if exact_region is not None:
            out[f"{region}_exact"] = exact_region
        pred_region_len = _as_float(_pred_value(pred, f"{region}_length", f"{metric}_length"))
        if pred_region_len is None and pred_seq is not None:
            pred_region_len = float(len(str(pred_seq)))
        truth_region_len = _as_float(record.get(f"{region}_length"))
        if pred_region_len is not None and truth_region_len is not None:
            out[f"{metric}_length_mae"] = abs(pred_region_len - truth_region_len)
            out[f"{region}_length_mae"] = abs(pred_region_len - truth_region_len)

    p_truth_parts = [
        _as_float(record.get(key)) or 0.0
        for key in ("p_v_3_length", "p_d_5_length", "p_d_3_length", "p_j_5_length")
    ]
    p_truth = sum(p_truth_parts)
    p_pred = _as_float(_pred_value(pred, "p_region_length"))
    if p_pred is None:
        p_pred_parts = [
            _as_float(_pred_value(pred, key)) or 0.0
            for key in ("p_v_3_length", "p_d_5_length", "p_d_3_length", "p_j_5_length")
        ]
        if any(key in pred for key in ("p_v_3_length", "p_d_5_length", "p_d_3_length", "p_j_5_length")):
            p_pred = sum(p_pred_parts)
    if p_pred is not None:
        out["p_region_length_mae"] = abs(p_pred - p_truth)
    return out


def _score_metadata(pred: dict[str, Any], case: BenchmarkCase) -> dict[str, float]:
    record = case.record
    out: dict[str, float] = {}

    pred_locus = _pred_value(pred, "locus", "chain_type")
    truth_locus = record.get("locus")
    if pred_locus is not None and truth_locus:
        out["locus_acc"] = 1.0 if str(pred_locus).upper() == str(truth_locus).upper() else 0.0
        out["chain_type_acc"] = out["locus_acc"]

    truth_has_d = bool(case.genes.get("d") and case.genes["d"].calls)
    if "d_call" in pred or "d_calls" in pred:
        out["has_d_routing_acc"] = 1.0 if bool(normalize_call_set(pred, "d")) == truth_has_d else 0.0

    pred_c = _pred_value(pred, "c_call", "constant_call")
    truth_c = record.get("c_call")
    if truth_c:
        out["c_found_rate"] = 1.0 if pred_c is not None else 0.0
        exact = _string_exact(pred_c, truth_c)
        if exact is not None:
            out["c_call_acc"] = exact
            out["c_gene_acc"] = 1.0 if str(pred_c).split("*")[0] == str(truth_c).split("*")[0] else 0.0

    pred_d_inv_raw = _pred_value(pred, "d_inverted", "d_orientation")
    pred_d_inv = _as_bool(pred_d_inv_raw)
    truth_d_inv = _as_bool(record.get("d_inverted"))
    if pred_d_inv is not None and truth_d_inv is not None:
        out["d_inversion_acc"] = 1.0 if pred_d_inv == truth_d_inv else 0.0

    for key, metric in (("vj_in_frame", "vj_in_frame_acc"), ("stop_codon", "stop_codon_acc")):
        pred_bool = _as_bool(_pred_value(pred, key))
        truth_bool = _as_bool(record.get(key))
        if pred_bool is not None and truth_bool is not None:
            out[metric] = 1.0 if pred_bool == truth_bool else 0.0

    for key, metric in (
        ("n_fwr1_mutations", "fwr1_mutation_count_mae"),
        ("n_cdr1_mutations", "cdr1_mutation_count_mae"),
        ("n_fwr2_mutations", "fwr2_mutation_count_mae"),
        ("n_cdr2_mutations", "cdr2_mutation_count_mae"),
        ("n_fwr3_mutations", "fwr3_mutation_count_mae"),
    ):
        pred_count = _as_float(_pred_value(pred, key, key.removeprefix("n_")))
        truth_count = _as_float(record.get(key))
        if pred_count is not None and truth_count is not None:
            out[metric] = abs(pred_count - truth_count)

    pred_layout = _pred_value(pred, "read_layout")
    truth_layout = record.get("read_layout")
    if pred_layout is not None and truth_layout:
        out["layout_specific_call_acc"] = 1.0 if str(pred_layout) == str(truth_layout) else 0.0

    truth_contaminant = _as_bool(record.get("is_contaminant"))
    pred_contaminant = _as_bool(_pred_value(pred, "is_contaminant", "contaminant"))
    if truth_contaminant:
        any_call = any(normalize_call_set(pred, gene) for gene in GENES)
        flagged = bool(pred_contaminant)
        out["contaminant_no_call_rate"] = 0.0 if any_call else 1.0
        # "handled" = the contaminant was identified, by EITHER flagging it OR no-calling it
        out["contaminant_handled_rate"] = 1.0 if (flagged or not any_call) else 0.0
        # a false-positive alignment is a confident call that was NOT flagged as contaminant
        out["false_positive_alignment_rate"] = 1.0 if (any_call and not flagged) else 0.0
    if pred_contaminant is not None and truth_contaminant is not None:
        out["contaminant_flag_acc"] = 1.0 if pred_contaminant == truth_contaminant else 0.0

    truth_revision = _as_bool(record.get("receptor_revision_applied"))
    pred_revision = _as_bool(_pred_value(pred, "receptor_revision_applied", "revision_applied"))
    if pred_revision is not None and truth_revision is not None:
        out["revision_flag_acc"] = 1.0 if pred_revision == truth_revision else 0.0
    if truth_revision:
        truth_v = case.genes.get("v")
        if truth_v and truth_v.calls:
            pred_v = normalize_call_set(pred, "v")
            out["revision_case_call_acc"] = 1.0 if pred_v and pred_v[0] in set(truth_v.calls) else 0.0
        boundary_errs = []
        for suffix in ("ss", "se"):
            p = _coord(pred, "v", suffix)
            t = truth_v.sequence_start if suffix == "ss" else truth_v.sequence_end
            if p is not None and t is not None:
                boundary_errs.append(abs(p - t))
        if boundary_errs:
            out["revision_case_boundary_mae"] = _avg(boundary_errs)

    return out


def score_one_case(
    case: BenchmarkCase,
    prediction: dict[str, Any] | None,
    *,
    frame: str = "canonical",
    include_expensive_record_fields: bool = True,
) -> dict[str, Any]:
    """Score one prediction against one benchmark case."""

    pred = prediction or {}
    global_metrics: dict[str, float] = {}
    gene_metrics: dict[str, dict[str, float]] = {}
    if "orientation_id" in pred and pred["orientation_id"] is not None:
        global_metrics["orientation_acc"] = (
            1.0 if int(pred["orientation_id"]) == int(case.orientation_id) else 0.0
        )
    global_metrics.update(_score_airr_contract(pred, case))
    global_metrics.update(_score_segment_order(pred, case, frame))
    global_metrics.update(_score_junction(pred, case, frame))
    global_metrics.update(_score_metadata(pred, case))
    global_metrics.update(prediction_performance_metrics(pred))
    for name in ("region", "state"):
        global_metrics.update(_score_labels(pred, case, name, frame))
    for scalar in ("noise_count", "mutation_rate", "indel_count", "productive"):
        global_metrics.update(_score_scalar(pred, case, scalar))
    alt_frame = "presented" if frame == "canonical" else "canonical"
    for gene, truth in case.truth(frame).items():
        gene_metrics[gene] = _score_gene(
            pred,
            truth,
            gene,
            case=case,
            alt_truth=case.truth(alt_frame).get(gene),
            include_expensive_record_fields=include_expensive_record_fields,
        )
    return {"global": global_metrics, "genes": gene_metrics}


def _score_cases(
    cases: Iterable[BenchmarkCase],
    predictions: Iterable[dict[str, Any]],
    *,
    frame: str = "canonical",
    include_strata: bool = True,
    include_expensive_record_fields: bool = True,
) -> dict[str, Any]:
    """Score predictions against benchmark cases.

    ``frame`` controls whether coordinates/labels are compared to canonical or
    presented-frame truth.
    """

    cases = list(cases)
    predictions = list(predictions)
    if len(cases) != len(predictions):
        raise ValueError(f"case/prediction length mismatch: {len(cases)} != {len(predictions)}")

    gene_metrics: dict[str, dict[str, list[float]]] = {g: defaultdict(list) for g in GENES}
    global_metrics: dict[str, list[float]] = defaultdict(list)
    by_stratum_cases: dict[str, list[BenchmarkCase]] = defaultdict(list)
    by_stratum_preds: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for case, pred in zip(cases, predictions):
        by_stratum_cases[case.stratum].append(case)
        by_stratum_preds[case.stratum].append(pred or {})
        one = score_one_case(
            case,
            pred,
            frame=frame,
            include_expensive_record_fields=include_expensive_record_fields,
        )
        for k, v in one["global"].items():
            global_metrics[k].append(v)
        for gene, vals in one["genes"].items():
            for k, v in vals.items():
                gene_metrics[gene][k].append(v)

    out = {
        "n_cases": len(cases),
        "frame": frame,
        "global": {k: _avg(v) for k, v in sorted(global_metrics.items())},
        "genes": {
            gene: {k: _avg(v) for k, v in sorted(vals.items())}
            for gene, vals in gene_metrics.items()
        },
    }
    if include_strata:
        out["by_stratum"] = {
            name: _score_cases(
                by_stratum_cases[name],
                by_stratum_preds[name],
                frame=frame,
                include_strata=False,
                include_expensive_record_fields=include_expensive_record_fields,
            )
            for name in sorted(by_stratum_cases)
        }
    else:
        out["by_stratum"] = {}
    return out


def score_cases(
    cases: Iterable[BenchmarkCase],
    predictions: Iterable[dict[str, Any]],
    *,
    frame: str = "canonical",
    include_strata: bool = True,
    include_expensive_record_fields: bool = True,
) -> dict[str, Any]:
    """Score predictions against benchmark cases."""

    return _score_cases(
        cases,
        predictions,
        frame=frame,
        include_strata=include_strata,
        include_expensive_record_fields=include_expensive_record_fields,
    )


def compact_summary(scores: dict[str, Any]) -> dict[str, Any]:
    """Return a small high-signal summary suitable for tables/logging."""

    genes = scores.get("genes", {})
    out = {"n_cases": scores.get("n_cases", 0), "frame": scores.get("frame")}
    for g in GENES:
        gm = genes.get(g, {})
        out[g] = {
            "call": gm.get("call_top1_in_set", math.nan),
            "set_f1": gm.get("call_set_f1", math.nan),
            "gene": gm.get("gene_top1_in_set", math.nan),
            "seq_mae": [gm.get("ss_mae", math.nan), gm.get("se_mae", math.nan)],
            "germ_mae": [gm.get("gs_mae", math.nan), gm.get("ge_mae", math.nan)],
            "segment_iou": gm.get("seq_span_iou", math.nan),
        }
    out["global"] = scores.get("global", {})
    return out
