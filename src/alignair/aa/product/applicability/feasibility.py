"""Split-aware feasibility for the Gate-3 cohort.

Turns hydrated, labelled `CohortEntry`s into a GO/STOP by: dedup by independent lineage -> allocate
fit/val/test -> require the **LOCKED TEST** slice of each *gated* group to meet its zero-error 95%
upper-bound floor. Powers the locked test, NOT the total cohort: a <=5% FAR upper bound needs >=59
independent locked-test examples, <=10% needs >=29, FRR<=10% needs >=29 locked-test positives; fit and
validation are ADDITIONAL. Zero-error floors — observed errors during real evaluation raise the
requirement (recompute with `min_n_for_upper_bound`).

Group-specific rules (from the settled decisions):
- Combined groups (`igk_igl`, `tcr`) require BOTH constituent loci present; a missing locus -> STOP.
- gamma/delta TCR (TRG/TRD) are DESCRIPTIVE — present but excluded from the TCR floor count.
- Structural-negative effective N counts PARENT lineages, not derived transforms (the dedup key is the
  parent lineage).
- Descriptive groups (de-novo) and descriptive strata are reported but never gate GO/STOP.
"""
from __future__ import annotations

import math
from collections import defaultdict

from .cohort import (MANIFEST, DESCRIPTIVE, GATED_5, GATED_10, POSITIVE, STRUCTURAL,
                     G_IGK_IGL, G_TCR, G_VHH_NONHUMAN, G_IGSF, G_STRUCTURAL,
                     G_POS_REAL, G_POS_SIM)


def min_n_zero_error(p: float, alpha: float = 0.05) -> int:
    """Smallest n whose zero-error one-sided (1-alpha) upper confidence bound is <= p.
    (1-p)**n <= alpha  ->  n >= ln(alpha)/ln(1-p)."""
    return math.ceil(math.log(alpha) / math.log(1 - p))


def min_n_for_upper_bound(p: float, errors: int, alpha: float = 0.05) -> int:
    """Smallest n whose exact (Clopper-Pearson) one-sided upper bound with `errors` observed failures
    is <= p. Zero-error reduces to `min_n_zero_error`. Used to re-power a group if the locked test
    turns up errors."""
    from scipy.stats import beta
    n = max(errors + 1, 1)
    while True:
        # one-sided upper bound = Beta(errors+1, n-errors).ppf(1-alpha)
        ub = beta.ppf(1 - alpha, errors + 1, n - errors) if n - errors > 0 else 1.0
        if ub <= p:
            return n
        n += 1


FLOOR_5 = min_n_zero_error(0.05)    # 59
FLOOR_10 = min_n_zero_error(0.10)   # 29

# Predeclared allocation policy. The LOCKED TEST slice must meet the floor; fit/val are additional.
POS_TEST_FRAC, POS_VAL_FRAC, POS_FIT_FRAC = 0.25, 0.25, 0.50   # positives: manifold fit needs the bulk
NEG_TEST_FRAC, NEG_VAL_FRAC = 0.50, 0.50                        # negatives: no fit (positive-only detector)

# group -> (floor, role)
GROUP_GATE = {
    G_IGK_IGL: (FLOOR_5, "neg"),
    G_TCR: (FLOOR_5, "neg"),
    G_VHH_NONHUMAN: (FLOOR_10, "neg"),
    G_IGSF: (FLOOR_10, "neg"),
    G_STRUCTURAL: (FLOOR_10, "neg"),
    G_POS_REAL: (FLOOR_10, "pos"),
    G_POS_SIM: (FLOOR_10, "pos"),
}
# combined groups that require >=1 deduped entry in EACH listed category (the TCR floor is carried by
# TRA/TRB; gamma/delta are descriptive and excluded from the floor but still must be reportable).
BOTH_LOCI = {G_IGK_IGL: {"igk", "igl"}, G_TCR: {"tra", "trb"}}

_CATEGORY_GATE = {spec.key: spec.gate for spec in MANIFEST}


def _required_effective_n(floor: int, role: str) -> int:
    """Effective (deduped) N needed so the test slice alone meets `floor` under the allocation policy."""
    test_frac = POS_TEST_FRAC if role == "pos" else NEG_TEST_FRAC
    return math.ceil(floor / test_frac)


def feasibility_report(entries) -> dict:
    """Compute per-group feasibility + an overall GO/STOP. `entries` is an iterable of `CohortEntry`."""
    # dedup by (group, lineage_key); first entry per lineage wins
    seen: dict = defaultdict(dict)              # group -> {lineage_key: entry}
    cats_by_group: dict = defaultdict(lambda: defaultdict(set))   # group -> category -> {lineage_key}
    for e in entries:
        if e.lineage_key not in seen[e.group]:
            seen[e.group][e.lineage_key] = e
        cats_by_group[e.group][e.category].add(e.lineage_key)

    groups: dict = {}
    stops: list = []
    for group, floor_role in GROUP_GATE.items():
        floor, role = floor_role
        deduped = list(seen.get(group, {}).values())
        # entries counting toward the floor = non-descriptive categories only
        floor_n = sum(1 for e in deduped if _CATEGORY_GATE.get(e.category) != DESCRIPTIVE)
        descriptive_n = len(deduped) - floor_n
        required = _required_effective_n(floor, role)

        both_ok, missing = True, []
        for cat in BOTH_LOCI.get(group, set()):
            if not cats_by_group[group].get(cat):
                both_ok = False
                missing.append(cat)

        powered = floor_n >= required and both_ok
        test_frac = POS_TEST_FRAC if role == "pos" else NEG_TEST_FRAC
        val_frac = POS_VAL_FRAC if role == "pos" else NEG_VAL_FRAC
        groups[group] = {
            "effective_after_dedup": len(deduped),
            "counts_toward_floor": floor_n,
            "descriptive_present": descriptive_n,
            "locked_test_floor": floor,
            "required_effective_n": required,
            "alloc_test": int(floor_n * test_frac),
            "alloc_val": int(floor_n * val_frac),
            "alloc_fit": int(floor_n * POS_FIT_FRAC) if role == "pos" else 0,
            "both_loci_present": both_ok,
            "missing_loci": missing,
            "powered": powered,
        }
        if not powered:
            reason = (f"missing loci {missing}" if not both_ok
                      else f"{floor_n} < required {required} (locked test needs >= {floor})")
            stops.append(f"{group}: {reason}")

    return {
        "floors": {"5pct": FLOOR_5, "10pct": FLOOR_10},
        "allocation_policy": {"pos": [POS_FIT_FRAC, POS_VAL_FRAC, POS_TEST_FRAC],
                              "neg": [0.0, NEG_VAL_FRAC, NEG_TEST_FRAC]},
        "groups": groups,
        "decision": "GO" if not stops else "STOP",
        "stops": stops,
    }
