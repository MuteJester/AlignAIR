"""Stage-4 Task 4.4 — compute budget planning + accounting for the full main+LOCO training program."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

BUDGET_PATH = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability/budget.json")


@dataclass
class Estimate:
    n_main_candidates: int
    seed_replicates: int
    main_seed_runs: int          # learned candidates x dev seed replicates (>= 3 each)
    chosen_main: int             # one frozen production checkpoint per candidate
    loco_models: int             # every learned LOCO model (candidate x fold)
    c0: int
    c0_profile_withheld: int     # C0 profile-withheld evaluations (per fold)
    total_models: int


def plan(candidates, seed_replicates, loco_folds) -> Estimate:
    n = len(candidates)
    main_seed_runs = n * seed_replicates
    loco_models = n * loco_folds
    total = main_seed_runs + n + loco_models + 1 + loco_folds
    return Estimate(n_main_candidates=n, seed_replicates=seed_replicates, main_seed_runs=main_seed_runs,
                    chosen_main=n, loco_models=loco_models, c0=1, c0_profile_withheld=loco_folds,
                    total_models=total)


def record(model_id, wall_s, gpu_h, params, path=BUDGET_PATH):
    data = json.loads(path.read_text()) if Path(path).exists() else {"runs": {}}
    data["runs"][model_id] = {"wall_s": float(wall_s), "gpu_h": float(gpu_h), "params": int(params)}
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True))


def total(path=BUDGET_PATH):
    data = json.loads(Path(path).read_text()) if Path(path).exists() else {"runs": {}}
    runs = data["runs"].values()
    return {"wall_s": sum(r["wall_s"] for r in runs), "gpu_h": sum(r["gpu_h"] for r in runs),
            "n_runs": len(data["runs"])}


def plan_dict(candidates, seed_replicates, loco_folds):
    return asdict(plan(candidates, seed_replicates, loco_folds))
