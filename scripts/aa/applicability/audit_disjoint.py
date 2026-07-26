"""CLI: build the verified pre-split Gate-3 denylist and audit a fresh cohort for disjointness.

Thin glue over the tested `alignair.aa.guard.disjoint` functions. Never reads the sealed partition.

Usage:
    python scripts/aa/applicability/audit_disjoint.py FRESH_COHORT.jsonl \
        [--cleaned .private/aa-gate3-hydration/cleaned_cohort.jsonl] \
        [--fingerprint docs/superpowers/artifacts/2026-07-24-aa-gate3/split_fingerprint.json]
"""
from __future__ import annotations

import argparse
import json
import sys

from alignair.aa.guard.disjoint import audit_disjoint, build_gate3_global_denylist

_DEF_CLEANED = ".private/aa-gate3-hydration/cleaned_cohort.jsonl"
_DEF_FP = "docs/superpowers/artifacts/2026-07-24-aa-gate3/split_fingerprint.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fresh_cohort")
    ap.add_argument("--cleaned", default=_DEF_CLEANED)
    ap.add_argument("--fingerprint", default=_DEF_FP)
    args = ap.parse_args(argv)

    fingerprint = json.load(open(args.fingerprint))
    denylist = build_gate3_global_denylist(args.cleaned, fingerprint)  # raises on digest mismatch
    fresh = [json.loads(line) for line in open(args.fresh_cohort) if line.strip()]

    report = audit_disjoint(fresh, gate3_denylist=denylist, phase3=({}, {}))
    print(json.dumps({
        "clean": report.clean,
        "overlap_counts": {k: len(v) for k, v in report.overlaps.items()},
        "denylist_sizes": {k: len(v) for k, v in denylist.items()},
    }, indent=2, sort_keys=True))
    return 0 if report.clean else 1


if __name__ == "__main__":
    sys.exit(main())
