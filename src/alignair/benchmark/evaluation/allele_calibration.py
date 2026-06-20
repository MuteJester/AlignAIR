"""Post-hoc calibration of the multi-label allele equivalence set (codex recipe).

The reader emits a per-candidate alignment score s_c (a length-normalized log-partition).
The equivalence SET is a temperature-scaled log-likelihood-ratio band:

    keep candidate c  iff  (s_top - s_c) / T <= epsilon

i.e. "the top is at most exp(epsilon) times likelier than c". This is invariant to the
candidate-independent per-read offset the state-conditioned emission introduces (it cancels
in s_top - s_c), so it is the stable object for "these alleles are not distinguishable".

T is fit per gene by multi-positive NLL on a LABELED calibration stream; epsilon is then
swept to the smallest mean set size achieving a target set-recall. Both are stored in a
JSON-able dict {GENE: {temperature, epsilon, ...}} that predict_reads(calibration=...) reads.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

# one calibration row per read/gene: (candidate scores, indices of truth alleles present in C)
Row = Tuple[Sequence[float], Sequence[int]]


def _logsumexp(xs: Sequence[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def multipos_nll(rows: Sequence[Row], T: float) -> float:
    """Mean multi-positive NLL: -log( sum_{c in truth} e^{s_c/T} / sum_{c} e^{s_c/T} ).
    Rows whose truth set is absent from the candidate list are skipped (top-k miss — a
    candidate-recall failure that no threshold can fix)."""
    total, n = 0.0, 0
    for scores, pos in rows:
        if not pos or not scores:
            continue
        s = [x / T for x in scores]
        denom = _logsumexp(s)
        numer = _logsumexp([s[j] for j in pos])
        total += -(numer - denom)
        n += 1
    return total / max(n, 1)


def fit_temperature(rows: Sequence[Row], grid: Sequence[float] | None = None) -> float:
    """Grid-search the per-gene temperature that minimizes multi-positive NLL."""
    grid = grid or [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    best_T, best = 1.0, float("inf")
    for T in grid:
        v = multipos_nll(rows, T)
        if v < best:
            best, best_T = v, T
    return best_T


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
            if rec >= target_recall and (best is None or size < best[1]):
                best = (eps, size, rec, f1)
        else:  # f1: maximize F1 with a recall floor
            if rec >= min_recall and (best is None or f1 > best[3]):
                best = (eps, size, rec, f1)
    eps, size, rec, f1 = best if best is not None else fallback
    return {"epsilon": eps, "mean_set_size": size, "set_recall": rec, "set_f1": f1,
            "objective": objective, "hit_target": best is not None}


def fit_calibration(per_gene_rows: Dict[str, Sequence[Row]], objective: str = "f1",
                    target_recall: float = 0.95, min_recall: float = 0.80) -> Dict[str, dict]:
    """Fit {GENE: {temperature, epsilon, mean_set_size, set_recall, set_f1, n, ...}}."""
    out: Dict[str, dict] = {}
    for G, rows in per_gene_rows.items():
        usable = [r for r in rows if r[1] and r[0]]
        T = fit_temperature(usable)
        sw = sweep_epsilon(usable, T, objective=objective,
                           target_recall=target_recall, min_recall=min_recall)
        out[G] = {"temperature": T, "n": len(usable),
                  "topk_truth_recall": len(usable) / max(len(rows), 1), **sw}
    return out


def collect_calibration_rows(model, reference_set, records, *, predict_fn=None,
                             topk: int = 32, genes=("v", "d", "j"), **predict_kwargs
                             ) -> Dict[str, List[Row]]:
    """Run the model over labeled `records` and build per-gene calibration rows from the
    emitted candidate scores + each record's comma-separated true call set."""
    if predict_fn is None:
        from ...inference.dnalignair_infer import predict_reads as predict_fn
    reads = [r["sequence"] for r in records]
    has_d = reference_set.has_d
    use_genes = [g for g in genes if g != "d" or has_d]
    preds = predict_fn(model, reference_set, reads, topk=topk, rerank="learned",
                       emit_scores=True, **predict_kwargs)
    rows: Dict[str, List[Row]] = {g.upper(): [] for g in use_genes}
    for rec, p in zip(records, preds):
        for g in use_genes:
            scored = p.get(f"{g}_scores")
            if not scored:
                continue
            names = [nm for nm, _ in scored]
            scores = [sc for _, sc in scored]
            truth = {c.strip() for c in str(rec.get(f"{g}_call", "")).split(",") if c.strip()}
            pos = [j for j, nm in enumerate(names) if nm in truth]
            rows[g.upper()].append((scores, pos))
    return rows
