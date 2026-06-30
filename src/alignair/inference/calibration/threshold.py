from __future__ import annotations

from typing import Dict, Sequence, Tuple

# one calibration row per read/gene: (candidate scores, indices of truth alleles present in C)
Row = Tuple[Sequence[float], Sequence[int]]


def _set_stats(rows: Sequence[Row], T: float, eps: float) -> Tuple[float, float, float]:
    """(mean set size, mean set recall, mean set F1) for an LR-band at temperature T,
    cutoff eps. Per-row recall = |kept∩pos|/|pos|, precision = |kept∩pos|/|kept|."""
    sizes, recalls, f1s = [], [], []
    for scores, pos in rows:
        if not pos or not scores:
            continue
        top = max(scores)
        kept = {j for j in range(len(scores)) if (top - scores[j]) / T <= eps}
        posset = set(pos)
        inter = len(kept & posset)
        rec = inter / len(posset)
        prec = inter / max(len(kept), 1)
        sizes.append(len(kept))
        recalls.append(rec)
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    n = max(len(sizes), 1)
    return sum(sizes) / n, sum(recalls) / n, sum(f1s) / n


def sweep_epsilon(rows: Sequence[Row], T: float, objective: str = "f1",
                  target_recall: float = 0.95, min_recall: float = 0.80,
                  grid: Sequence[float] | None = None) -> Dict[str, float]:
    """Pick epsilon by `objective`:
      'f1'        -> maximize mean set-F1 subject to set-recall >= min_recall (balanced;
                     avoids the recall-only blow-up that bloats sets on easy data);
      'recall'    -> smallest mean set size whose set-recall >= target_recall (legacy).
    Falls back to the most permissive epsilon (max recall) if no point qualifies."""
    grid = grid or [round(0.1 * i, 2) for i in range(0, 81)]   # 0.0 .. 8.0
    best = None
    fallback = None
    for eps in grid:
        size, rec, f1 = _set_stats(rows, T, eps)
        fallback = (eps, size, rec, f1)
        if objective == "recall":
            if rec >= target_recall and (best is None or size < best[1]): # Wait, "parks and"? No, the original code had: "if rec >= target_recall and (best is None or size < best[1]):"
                best = (eps, size, rec, f1)
        else:  # f1: maximize F1 with a recall floor
            if rec >= min_recall and (best is None or f1 > best[3]):
                best = (eps, size, rec, f1)
    eps, size, rec, f1 = best if best is not None else fallback
    return {"epsilon": eps, "mean_set_size": size, "set_recall": rec, "set_f1": f1,
            "objective": objective, "hit_target": best is not None}
