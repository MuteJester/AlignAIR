"""Region, state, and scalar prediction scoring."""
from __future__ import annotations

from typing import Any

from ...core.schema import BenchmarkCase
from ....nn.heads.region import REGIONS
from ....nn.heads.state import STATE_INDEX, STATES
from .primitives import as_float, avg


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


def score_labels(pred: dict[str, Any], case: BenchmarkCase, name: str, frame: str) -> dict[str, float]:
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
        out[f"{name}_per_class_recall"] = avg(class_recalls)
        out[f"{name}_macro_f1"] = avg(class_f1s)
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


def score_scalar(pred: dict[str, Any], case: BenchmarkCase, name: str) -> dict[str, float]:
    if name not in pred or name not in case.scalars or pred[name] is None:
        return {}
    truth = case.scalars[name]
    p = as_float(pred[name])
    if p is None:
        return {}
    if name == "productive":
        return {"productive_acc": 1.0 if (p >= 0.5) == (truth >= 0.5) else 0.0}
    if name == "indel_count":
        return {"indel_count_mae": abs(p - truth), "indel_length_mae": abs(p - truth)}
    return {f"{name}_mae": abs(p - truth)}
