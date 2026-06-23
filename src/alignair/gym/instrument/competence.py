"""CompetenceMetric: the single pre-registered EXTERNAL deployment-alignment score
(NOT the Kendall training loss), so it is comparable across architectures. A weighted
composite of allele calls, coordinate accuracy (within a fixed nt tolerance), region
accuracy, and junction exact-match. Aggregated with bootstrap CIs."""
from typing import Sequence

from .stats import bootstrap_ci

_DEFAULT_WEIGHTS = {"v_call": 0.2, "d_call": 0.1, "j_call": 0.15,
                    "coords": 0.25, "region": 0.15, "junction": 0.15}

# sub-metric name -> the record key that supplies it
_KEY = {"v_call": "v_call_correct", "d_call": "d_call_correct", "j_call": "j_call_correct",
        "coords": "coord_errs", "region": "region_acc", "junction": "junction_exact"}


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
        # Renormalize over the sub-metrics actually PRESENT in the record. A sub-metric
        # the evaluator didn't emit (e.g. junction before it's wired, or D on a row where
        # D is deliberately unsupervised) is EXCLUDED — not scored 0 — so it neither caps
        # competence nor biases pacing. Comparability across architectures holds as long
        # as they emit the same sub-metrics.
        num, wsum = 0.0, 0.0
        for name, w in self.weights.items():
            key = _KEY.get(name)
            if key is None or key not in rec:
                continue
            val = self._coord_subscore(rec[key]) if name == "coords" else float(rec[key])
            num += w * val
            wsum += w
        return num / wsum if wsum > 0 else 0.0

    def aggregate(self, recs: Sequence[dict], seed: int = 0) -> dict:
        scores = [self.score(r) for r in recs]
        mean, lo, hi = bootstrap_ci(scores, seed=seed)
        return {"S": mean, "lo": lo, "hi": hi, "n": len(scores)}
