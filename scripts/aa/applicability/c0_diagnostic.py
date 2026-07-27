"""Stage-2 Step 6 — C0 development diagnostic on the burned Gate-3 FIT split ONLY.

This is a DEVELOPMENT DIAGNOSTIC, not fresh evidence and not a release PASS. C0's rule structure, score
directions, and finite threshold grid are frozen on burned development data; the single grid point is chosen
on the fresh selection subset later (Stage 6). This script NEVER reads fresh selection, the blind-LOCO
challenge, or the sealed test (locked_test.jsonl). It reads only dev_cohort.jsonl records with split=='fit'
(the sealed test is a separate file that is never opened).

Run:  PYTHONPATH=src .venv/bin/python scripts/aa/applicability/c0_diagnostic.py
"""
from __future__ import annotations

import collections
import datetime
import json
from pathlib import Path

from alignair.aa.guard.c0_rules import C0_THRESHOLDS, c0_decide
from alignair.aa.guard.features import hmm_features
from alignair.aa.guard.profiles import load_bank

DEV = Path(".private/aa-gate3-hydration/dev_cohort.jsonl")     # fit+val only; the sealed test is NOT here
PROF = Path(".private/aa-applicability/profiles")
ART = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability")


def main():
    assert DEV.exists(), "dev_cohort.jsonl missing"
    fit = [json.loads(l) for l in open(DEV) if json.loads(l).get("split") == "fit"]
    assert fit and all(r["split"] == "fit" for r in fit), "must be fit-only"
    bank = load_bank(PROF)

    by_state = collections.Counter()
    by_cat = collections.defaultdict(lambda: collections.Counter())
    miss_reasons = collections.Counter()
    for r in fit:
        f = hmm_features(bank, r["aa"])
        v = c0_decide(f)
        by_state[v.state] += 1
        by_cat[r["category"]][v.state] += 1
        if v.state != "supported":
            miss_reasons[f"{v.state}:{v.reason}"] += 1

    n = len(fit)
    supported = by_state["supported"]
    diag = {
        "artifact": "c0_dev_diagnostic",
        "kind": "development diagnostic (burned Gate-3 FIT split only) — NOT fresh evidence, NOT a release PASS",
        "computed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seal_safety": "reads dev_cohort.jsonl split=='fit' only; locked_test.jsonl (seal) never opened; "
                       "fresh selection / blind-LOCO never read",
        "fit_n": n,
        "expected_state": "supported (all 240 fit records are positive human IGH VH)",
        "c0_state_counts": dict(by_state),
        "supported_recall_on_fit": round(supported / n, 4),
        "misclassification_reasons": dict(miss_reasons),
        "by_category": {c: dict(s) for c, s in by_cat.items()},
        "c0_thresholds_frozen": {k: (sorted(v) if isinstance(v, frozenset) else v)
                                 for k, v in C0_THRESHOLDS.items()},
        "note": "thresholds/grid frozen on burned-dev; the production grid point is selected on FRESH data "
                "in Stage 6, never here.",
    }
    (ART / "c0_dev_diagnostic.json").write_text(json.dumps(diag, indent=2, sort_keys=True))
    print(json.dumps({k: diag[k] for k in ("fit_n", "c0_state_counts", "supported_recall_on_fit",
                                           "misclassification_reasons")}, indent=2, sort_keys=True))
    return diag


if __name__ == "__main__":
    main()
