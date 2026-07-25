"""Stage-2 fit + VALIDATION selection (Tasks 6/9). Fits the positive-only completeness and membership
scorers on their endpoint-specific fit roles, then selects the scorer + operating threshold on VALIDATION
ONLY, by the deterministic rule: satisfy the confidence-bounded FAR constraints first, then minimize FRR
(upper bound), ties -> higher threshold (lower FAR). Stops before the locked test if nothing qualifies.

FAR/FRR are one-sided exact 95% upper bounds (Clopper-Pearson). Structural completeness negatives are
PARENT-LINEAGE aggregated (a parent is false-accepted if ANY of its transforms is accepted). The locked
test is never loaded here.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import beta

from alignair.aa.product.applicability import embedding, roles, scorers
from alignair.aa.product.applicability.cohort_loader import GateCohort
from alignair.aa.product.applicability.completeness import structural_prefilter

ART = "docs/superpowers/artifacts/2026-07-24-aa-gate3"
FRR_BOUND = 0.10
COMPLETENESS_FAR = {"structural": 0.10}
MEMBERSHIP_FAR = {"igk_igl": 0.05, "tcr": 0.05, "vhh_nonhuman": 0.10, "igsf": 0.10}
_MEMB_NEG_GROUP = {"igk": "igk_igl", "igl": "igk_igl", "tra": "tcr", "trb": "tcr", "trg": "tcr", "trd": "tcr",
                   "murine_vh": "vhh_nonhuman", "vhh": "vhh_nonhuman", "chimeric_murine_variable": "vhh_nonhuman",
                   "igsf_vset": "igsf"}


def cp_upper(k: int, n: int, alpha: float = 0.05) -> float:
    if n == 0:
        return 1.0
    return float(beta.ppf(1 - alpha, k + 1, n - k)) if n - k > 0 else 1.0


def _embed_all(predictor, recs, batch=64):
    feats = {}
    for i in range(0, len(recs), batch):
        chunk = recs[i:i + batch]
        f = embedding.branch_features(predictor, [r["aa"] for r in chunk])
        for r, v in zip(chunk, f):
            feats[r["id"]] = v
    return feats


def select(pos_scores, neg_groups, far_bounds):
    """Deterministic threshold selection on validation. Returns dict or None (no feasible threshold)."""
    alls = [pos_scores] + [v for v in neg_groups.values() if len(v)]
    cand = np.unique(np.concatenate(alls))
    best = None
    for tau in cand:
        fars = {}
        feasible = True
        for g, bound in far_bounds.items():
            s = neg_groups.get(g, np.array([]))
            ub = cp_upper(int((s >= tau).sum()), len(s))
            fars[g] = ub
            if ub > bound:
                feasible = False
        if not feasible:
            continue
        frr = cp_upper(int((pos_scores < tau).sum()), len(pos_scores))
        if frr > FRR_BOUND:
            continue
        key = (round(frr, 6), -float(tau))                    # min FRR, ties -> higher tau (lower FAR)
        if best is None or key < best["key"]:
            best = {"key": key, "threshold": float(tau), "far_upper": fars, "frr_upper": frr}
    return best


def run():
    import alignair
    predictor = alignair.load_aa_model("alignair-aa-igh-human@2.0.0")
    dev = GateCohort().dev()
    role = roles.assign_roles(dev)["roles"]
    by_id = {r["id"]: r for r in dev}
    # Stage-2 only scores what Stage-1 DEFERS; a Stage-1-rejected record is never "accepted"
    deferred = {r["id"]: not structural_prefilter(r["aa"]).rejected for r in dev}
    def_recs = [r for r in dev if deferred[r["id"]]]
    feats = _embed_all(predictor, def_recs)               # deferred are all <= MAX_COMPLETE -> fit in max_len

    report = {"frr_bound": FRR_BOUND, "note": "accept = Stage-1-defer AND Stage-2-score>=tau; "
              "Stage-1-rejected records score -inf (never accepted).", "endpoints": {}}
    for ep, fit_role, val_role, far_bounds in [
            ("completeness", "completeness_fit", "completeness_validation", COMPLETENESS_FAR),
            ("membership", "membership_fit", "membership_validation", MEMBERSHIP_FAR)]:
        fit_ids = [r["id"] for r in dev if role[r["id"]][ep] == fit_role and deferred[r["id"]]]
        Xfit = np.array([feats[i] for i in fit_ids])

        pos_ids, neg_ids = [], []
        for r in dev:
            if role[r["id"]][ep] != val_role:
                continue
            cat = r["category"]
            is_neg = (cat in _MEMB_NEG_GROUP) if ep == "membership" else (cat in roles._STRUCTURAL)
            (neg_ids if is_neg else pos_ids).append(r["id"])

        best_overall, diag = None, {}
        for name, Cls in scorers.CANDIDATES.items():
            sc = Cls().fit(Xfit)

            def score_of(rid):
                return float(sc.score(feats[rid][None, :])[0]) if deferred[rid] else -np.inf

            pos_scores = np.array([score_of(i) for i in pos_ids])
            if ep == "completeness":               # PARENT-aggregate structural negatives (max over transforms)
                per_parent = defaultdict(lambda: -np.inf)
                for i in neg_ids:
                    per_parent[by_id[i]["lineage_key"]] = max(per_parent[by_id[i]["lineage_key"]], score_of(i))
                neg_groups = {"structural": np.array(list(per_parent.values()))}
            else:
                grp = defaultdict(list)
                for i in neg_ids:
                    grp[_MEMB_NEG_GROUP[by_id[i]["category"]]].append(score_of(i))
                neg_groups = {g: np.array(v) for g, v in grp.items()}
            # diagnostic: separation AUC per group + the best FAR upper bound achievable at FRR_ub<=10%
            tau10 = np.quantile(pos_scores, 0.0)                 # start; find the highest tau with FRR_ub<=10%
            taus = np.unique(pos_scores)
            feas = [t for t in taus if cp_upper(int((pos_scores < t).sum()), len(pos_scores)) <= FRR_BOUND]
            tau_at_frr10 = max(feas) if feas else float(pos_scores.min())
            diag[name] = {g: {"n": int(len(v)),
                              "auc": round(float((pos_scores[:, None] > v[None, :]).mean()), 3),
                              "far_ub_at_frr10": round(cp_upper(int((v >= tau_at_frr10).sum()), len(v)), 3)}
                          for g, v in neg_groups.items()}
            chosen = select(pos_scores, neg_groups, far_bounds)
            if chosen and (best_overall is None or chosen["key"] < best_overall["key"]):
                best_overall = {**chosen, "scorer": name,
                                "n_fit": len(fit_ids), "n_val_pos": len(pos_ids),
                                "n_val_neg_groups": {g: int(len(v)) for g, v in neg_groups.items()}}
        report["endpoints"][ep] = best_overall or {
            "selected": None, "reason": "no feasible threshold on validation",
            "far_bounds": far_bounds, "separation_diagnostic": diag}

    both = all(report["endpoints"][ep].get("threshold") is not None for ep in ("completeness", "membership"))
    report["decision"] = "PROCEED_TO_LOCKED_TEST" if both else "STOP_NO_FEASIBLE_THRESHOLD"
    os.makedirs(ART, exist_ok=True)
    json.dump(report, open(os.path.join(ART, "stage2_selection.json"), "w"), indent=2, sort_keys=True, default=str)
    print("DECISION:", report["decision"])
    for ep, r in report["endpoints"].items():
        if r.get("threshold") is not None:
            print(f"  {ep:12s} scorer={r['scorer']:22s} thr={r['threshold']:.3f} "
                  f"FRR_ub={r['frr_upper']:.3f} FAR_ub={ {g: round(v,3) for g,v in r['far_upper'].items()} }")
        else:
            print(f"  {ep:12s} {r}")
    return report


if __name__ == "__main__":
    run()
