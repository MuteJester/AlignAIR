from __future__ import annotations

from typing import Dict, Sequence, Tuple

from .temperature import fit_temperature, Row
from .threshold import sweep_epsilon


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


def fit_contaminant_tau(real_gate_scores: Sequence[float], fpr_target: float = 0.02) -> float | None:
    """Out-of-scope gate threshold: tau = the `fpr_target` quantile of REAL reads' gate
    scores, so the worst `fpr_target` fraction of real reads fall below it (flagged). A read
    scoring below tau is flagged is_contaminant (flag-only). Returns None if no scores."""
    s = sorted(x for x in real_gate_scores if x is not None)
    if not s:
        return None
    idx = min(len(s) - 1, max(0, int(len(s) * fpr_target)))
    return float(s[idx])
