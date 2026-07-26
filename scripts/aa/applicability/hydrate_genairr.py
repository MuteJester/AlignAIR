"""Source (positive_sim) hydration: GenAIRR simulated human IGH VH positives.

Generates productive, complete VH amino-acid domains from GenAIRR HUMAN_IGH_OGRDB with a FRESH held-out
seed range (>= 1_100_000_000): held out from model training (seeds < 1e9) and disjoint from the Gate-3
simulator seeds (~1e9). Completeness/locus here is CONSTRUCTION TRUTH from the simulator (the independent
basis for simulated positives), not the guard. Denylist-filtered; counts-only census.

Run:  PYTHONPATH=src .venv/bin/python scripts/aa/applicability/hydrate_genairr.py
"""
from __future__ import annotations

import json
from pathlib import Path

from alignair.aa.guard.disjoint import build_gate3_global_denylist
from alignair.aa.guard.hydration import atomic_write_text, filter_disjoint, sha256_text

STORE = Path(".private/aa-applicability/fresh_hydration")
ART = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability/fresh_hydration")
CLEANED = ".private/aa-gate3-hydration/cleaned_cohort.jsonl"
FP = "docs/superpowers/artifacts/2026-07-24-aa-gate3/split_fingerprint.json"

START_SEED = 1_100_000_000   # >= 1e9 (held out from training) and above the reserved held-out block + Gate-3
BUDGET = 4000                # consecutive seeds scanned; ample to collect WANT productive-complete
WANT = 120                   # >= floor 36/subset with margin
RATE = 0.05                  # S5F SHM rate (matches training realism)


def _first(call):
    return (call or "").split(",")[0]


def main():
    STORE.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    denylist = build_gate3_global_denylist(CLEANED, json.load(open(FP)))

    import GenAIRR as ga
    exp = ga.Experiment.on("HUMAN_IGH_OGRDB").recombine().mutate(model="s5f", rate=RATE)

    records, generated, rejected = [], 0, {"not_productive": 0, "no_aa": 0, "stop_codon": 0,
                                           "no_frame": 0, "bad_len": 0}
    for i, rec in enumerate(exp.stream_records(n=BUDGET, seed=START_SEED)):
        generated += 1
        seed = START_SEED + i
        if not rec.get("productive"):
            rejected["not_productive"] += 1
            continue
        aa = rec.get("sequence_aa")
        if not aa:
            rejected["no_aa"] += 1
            continue
        if "*" in aa:
            rejected["stop_codon"] += 1
            continue
        if rec.get("junction_start") is None:
            rejected["no_frame"] += 1
            continue
        if not (100 <= len(aa) <= 160):
            rejected["bad_len"] += 1
            continue
        records.append({"id": f"pos_sim_{seed}", "aa": aa.upper(), "group": "positive_sim",
                        "category": "pos_igh_simulator", "lineage_key": f"genairr_fresh_seed_{seed}",
                        "seed": seed, "v_call": _first(rec.get("v_call")), "j_call": _first(rec.get("j_call")),
                        "source": "genairr_sim", "verification": "simulator_construction_truth"})
        if len(records) >= WANT:
            break

    kept, dropped = filter_disjoint(records, denylist=denylist, phase3=({}, {}))
    atomic_write_text(STORE / "genairr_sim.normalized.json", json.dumps(kept))

    lens = [len(r["aa"]) for r in kept]
    census = {
        "source": "genairr_sim",
        "provenance": {"dataconfig": "HUMAN_IGH_OGRDB", "engine": "GenAIRR recombine+s5f",
                       "rate": RATE, "seed_range": [START_SEED, START_SEED + generated - 1],
                       "held_out": "seeds >= 1e9 (training used < 1e9); disjoint from Gate-3 sim (~1e9)",
                       "completeness": "simulator construction truth (independent of guard)"},
        "generated_scanned": generated,
        "rejected": rejected,
        "kept_after_denylist": len(kept),
        "denylist_dropped": dropped,
        "distinct_seeds": len({r["seed"] for r in kept}),
        "aa_len": {"min": min(lens, default=0), "max": max(lens, default=0)},
        "distinct_v_calls": len({r["v_call"] for r in kept}),
        "distinct_j_calls": len({r["j_call"] for r in kept}),
    }
    atomic_write_text(ART / "genairr_sim.census.json", json.dumps(census, indent=2, sort_keys=True))
    print(json.dumps(census, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
