"""Test-only helpers shared by the assay's tests; no package module imports this one.

The oracle prediction lives in ``truth.oracle`` and the AIRR TSV writer in ``airr.writer``, since runners use them
too.
"""
from __future__ import annotations

from alignair_assay.core.reference import Reference
from alignair_assay.metrics import CaseScores
from alignair_assay.scoring.oracle import oracle_problems
from alignair_assay.scoring.pipeline import score_predictions
from alignair_assay.truth.model import ObservableTruth
from alignair_assay.truth.oracle import oracle_prediction

_MAX_LISTED = 10


def assert_oracle_perfect(truths: list[ObservableTruth], reference: Reference) -> dict[str, CaseScores]:
    """Assert the oracle rule (``scoring.oracle.oracle_problems``) on ``truths``: scored through
    ``scoring.pipeline.score_predictions``, the oracle prediction gets 1.0 on every metric that is neither
    lower-is-better nor informational, 0.0 on every lower-is-better metric, and no ``nan`` anywhere. At least one
    case must be scored; the failure names the first offending (case, metric, value) triples. Returns the oracle's
    per-case scores, so callers can check which metrics a suite emits."""
    preds = {t.case_id: oracle_prediction(t) for t in truths if t.excluded is None}
    scores = score_predictions(truths, preds, reference)
    assert scores, "no scored case: every truth is excluded"
    problems = oracle_problems(scores)
    assert not problems, f"{len(problems)} oracle scores are not perfect, e.g. {problems[:_MAX_LISTED]}"
    return scores
