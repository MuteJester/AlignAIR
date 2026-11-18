"""The oracle rule: what the oracle prediction must score on every suite, checked on per-case scores."""
from __future__ import annotations

import math

from alignair_assay.metrics import CaseScores, is_informational, lower_is_better


def oracle_problems(scores: dict[str, CaseScores]) -> list[tuple[str, str, float]]:
    """The ``(case id, metric, value)`` triples, in case and metric order, where the oracle's per-case ``scores``
    break the oracle rule: every metric that is neither lower-is-better nor informational must be 1.0, every
    lower-is-better metric 0.0, and no value may be ``nan``. An empty list means the oracle is perfect (when at
    least one case was scored); a suite that has problems has a truth defect."""
    problems = []
    for case_id, case in scores.items():
        for name, value in case.items():
            if math.isnan(value):
                problems.append((case_id, name, value))
            elif not is_informational(name) and value != (0.0 if lower_is_better(name) else 1.0):
                problems.append((case_id, name, value))
    return problems
