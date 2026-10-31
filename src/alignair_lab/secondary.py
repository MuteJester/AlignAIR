"""Secondary metrics: what the five headline metrics cannot see, read from the assay's cached per-suite diagnostics of
each model (its ``suite.json``: the model's own score per suite), never diagnosed again.

``METRIC``, ``d_short.abstain_ok``: on the reads whose D is too short to call, whether the call abstains. A D presence
gate can only lower D top-1 there, never raise it: its benefit is this abstention, which the headline metrics miss.

``cached_comparison`` shapes two models' scores as the assay shapes a comparison summary, so ``noise.decide`` weighs
them as it weighs the headline metrics: per suite both score on the same reads, the candidate's mean less the
baseline's. It has no read-level verdict (the assay pairs none for this metric), so a fall in a suite where the
baseline seeds have no spread is listed, never counted (``noise.suite_tests``). ``merged`` adds such a comparison to
the assay's. ``cached_seed_values`` values a calibration's seeds from its own cached diagnostics, for a calibration
whose pairs predate the metric (``baselines.calibration_seeds``).
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from alignair_lab.noise import pair, seed_values

METRIC = "d_short.abstain_ok"
SUITE_RECORD = "suite.json"               # a model's cached diagnostics of one suite, under <diagnostics>/<key>/<suite>
VARIANT = "baseline"                      # the end-to-end variant, whose scores the metric is


def suite_scores(diagnostics: str | Path, model_key: str, metric: str) -> dict[str, dict]:
    """Each suite of the model ``model_key``'s cached diagnostics under ``diagnostics`` (an assay run's
    ``diagnostics`` directory) -> its own ``metric`` score, ``mean`` and ``n``; suites that score it on no read, or
    whose record cannot be read, left out."""
    root = Path(diagnostics) / model_key
    found = {}
    for path in sorted(root.glob(f"*/{SUITE_RECORD}")) if root.is_dir() else []:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        entry = (((record.get("variants") or {}).get(VARIANT) or {}).get("scores") or {}).get(metric)
        if entry and entry.get("n") and entry.get("mean") is not None:
            found[path.parent.name] = {"mean": entry["mean"], "n": entry["n"]}
    return found


def cached_comparison(diagnostics: str | Path, candidate_key: str, baseline_key: str, metric: str) -> dict:
    """``metric``'s comparison of two models, shaped as the assay's summary (``suites`` -> suite -> ``headline`` ->
    metric -> ``delta``, ``n_paired``, ``test``, ``verdict``): per suite both score on the same number of reads, the
    candidate's mean less the baseline's, with no test and no verdict."""
    candidate, baseline = (suite_scores(diagnostics, key, metric) for key in (candidate_key, baseline_key))
    suites = {}
    for name, found in candidate.items():
        other = baseline.get(name)
        if other is not None and other["n"] == found["n"]:
            suites[name] = {"headline": {metric: {"delta": found["mean"] - other["mean"], "n_paired": found["n"],
                                                  "test": None, "verdict": None}}}
    return {"suites": suites}


def merged(comparison: Mapping, extra: Mapping | None) -> Mapping:
    """``comparison`` (an assay summary) with ``extra``'s entries (``cached_comparison``) added to the headline of each
    of its suites; new dicts, neither changed. ``comparison`` itself without ``extra``."""
    if not extra:
        return comparison
    suites = {}
    for name, suite in comparison["suites"].items():
        added = (((extra.get("suites") or {}).get(name) or {}).get("headline")) or {}
        suites[name] = {**suite, "headline": {**(suite.get("headline") or {}), **added}}
    return {**comparison, "suites": suites}


def cached_seed_values(diagnostics: str | Path, keys: Mapping[str, str], seeds: Sequence[int], *, relative_to: int,
                       metric: str) -> dict | None:
    """``metric``'s values of a calibration's ``seeds`` (``noise.seed_values``) relative to its screen of
    ``relative_to``, from their cached diagnostics under ``diagnostics`` (``keys``: seed -> the model key of its
    screen): each seed's ``cached_comparison`` against that screen. None when a seed has no key or no comparable
    scores."""
    if any(not keys.get(str(seed)) for seed in seeds):
        return None
    reference = keys[str(relative_to)]
    pairs = []
    for seed in seeds:
        if seed == relative_to:
            continue
        comparison = cached_comparison(diagnostics, keys[str(seed)], reference, metric)
        if not comparison["suites"]:
            return None
        pairs.append({"a": relative_to, "b": seed, **pair(comparison)})
    return seed_values(pairs, seeds, relative_to=relative_to)
