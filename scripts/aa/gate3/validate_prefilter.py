"""Validate the Stage-1 structural prefilter (binding safeguard): every hard rule must reject ~0 real +
simulator positives, else it is demoted to defer-only. Uses the sealed DEV split only. Writes a
committable validation record (counts only, no sequences)."""
from __future__ import annotations

import json
import os
from collections import Counter

from alignair.aa.product.applicability.completeness import structural_prefilter, RULESET_VERSION
from alignair.aa.product.applicability.cohort_loader import GateCohort

ART = "docs/superpowers/artifacts/2026-07-24-aa-gate3"
POSITIVES = ("pos_igh_repertoire", "pos_igh_simulator", "design_human_framework", "humanized",
             "pos_igh_structure")
STRUCTURAL = ("reversed_vh", "v_only", "cdr3_only", "j_only", "leader_vh", "vh_constant")


def run():
    dev = GateCohort().dev()
    by = Counter((r["category"], structural_prefilter(r["aa"]).reason or "DEFER") for r in dev)

    pos_rejects_by_rule = Counter()
    pos = {}
    for cat in POSITIVES:
        outs = {k[1]: v for k, v in by.items() if k[0] == cat}
        rej = {k: v for k, v in outs.items() if k != "DEFER"}
        for k, v in rej.items():
            pos_rejects_by_rule[k] += v
        pos[cat] = {"n": sum(outs.values()), "deferred": outs.get("DEFER", 0), "hard_rejects": rej}

    struct = {cat: {k[1]: v for k, v in by.items() if k[0] == cat} for cat in STRUCTURAL}

    # every hard rule must reject 0 positives to remain a hard rule
    hard_rules = ["too_short", "reversed_order", "leader_or_constant_contamination"]
    rule_status = {rule: {"positive_rejects": pos_rejects_by_rule.get(rule, 0),
                          "status": "HARD" if pos_rejects_by_rule.get(rule, 0) == 0 else "DEMOTE_TO_DEFER"}
                   for rule in hard_rules}

    record = {
        "ruleset_version": RULESET_VERSION,
        "design": "high-specificity prefilter: hard-reject decisive defects only; defer everything else "
                  "(mutated Cys/FR4 never reject) to Stage-2. Documents the superseded rule-gate's ~12% "
                  "FRR ceiling (commit 1a2e87a) as the reason motif presence cannot be mandatory.",
        "positive_frr": {"total_hard_rejects": sum(pos_rejects_by_rule.values()),
                         "by_rule": dict(pos_rejects_by_rule), "by_category": pos},
        "hard_rule_status": rule_status,
        "structural_negative_prefilter": struct,
        "verdict": "PASS" if sum(pos_rejects_by_rule.values()) == 0 else "FAIL",
    }
    os.makedirs(ART, exist_ok=True)
    json.dump(record, open(os.path.join(ART, "prefilter_validation.json"), "w"), indent=2, sort_keys=True)
    print(f"prefilter positive hard-rejects: {record['positive_frr']['total_hard_rejects']} -> "
          f"verdict {record['verdict']}")
    for rule, st in rule_status.items():
        print(f"  {rule:34s} positive_rejects={st['positive_rejects']} -> {st['status']}")
    return record


if __name__ == "__main__":
    run()
