"""CompetenceMetric: the single pre-registered EXTERNAL deployment-alignment score
(NOT the Kendall training loss), so it is comparable across architectures. A weighted
composite of allele calls, coordinate accuracy (within a fixed nt tolerance), region
accuracy, and junction exact-match. Aggregated with bootstrap CIs."""
from typing import Sequence

from .stats import bootstrap_ci

_DEFAULT_WEIGHTS = {"v_call": 0.2, "d_call": 0.1, "j_call": 0.15,
                    "coords": 0.25, "region": 0.15, "junction": 0.15}


class CompetenceMetric:
    def __init__(self, weights: dict | None = None, coord_tol: float = 2.0):
        self.weights = dict(weights) if weights is not None else dict(_DEFAULT_WEIGHTS)
        self.coord_tol = coord_tol

    def _coord_subscore(self, errs) -> float:
        errs = list(errs or [])
        if not errs:
            return 0.0
        within = sum(1 for e in errs if abs(e) <= self.coord_tol)
        return within / len(errs)

    def score(self, rec: dict) -> float:
        parts = {
            "v_call": float(rec.get("v_call_correct", 0)),
            "d_call": float(rec.get("d_call_correct", 0)),
            "j_call": float(rec.get("j_call_correct", 0)),
            "coords": self._coord_subscore(rec.get("coord_errs")),
            "region": float(rec.get("region_acc", 0.0)),
            "junction": float(rec.get("junction_exact", 0)),
        }
        wsum = sum(self.weights.get(k, 0.0) for k in parts)
        if wsum == 0:
            return 0.0
        return sum(self.weights.get(k, 0.0) * v for k, v in parts.items()) / wsum

    def aggregate(self, recs: Sequence[dict], seed: int = 0) -> dict:
        scores = [self.score(r) for r in recs]
        mean, lo, hi = bootstrap_ci(scores, seed=seed)
        return {"S": mean, "lo": lo, "hi": hi, "n": len(scores)}
