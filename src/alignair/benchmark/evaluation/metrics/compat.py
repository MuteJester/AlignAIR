"""Compatibility facade for benchmark scoring APIs.

New code should import from :mod:`alignair.benchmark.evaluation.scoring`.
"""

from ..scoring import compact_summary, score_cases, score_one_case

__all__ = ["compact_summary", "score_cases", "score_one_case"]
