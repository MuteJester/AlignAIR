"""Compute estimates and per-run accounting for guard training studies."""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Estimate:
    n_main_candidates: int
    seed_replicates: int
    main_seed_runs: int
    chosen_main: int
    loco_models: int
    c0: int
    c0_profile_withheld: int
    total_models: int


def plan(candidates: Sequence[str], seed_replicates: int, loco_folds: int) -> Estimate:
    """Count seed runs, chosen checkpoints, fold models, and C0 evaluations.

    ``total_models`` includes chosen checkpoints as separate accounting entries,
    matching the existing study reports; it is not a count of unique training runs.
    """
    n = len(candidates)
    main_seed_runs = n * seed_replicates
    loco_models = n * loco_folds
    total = main_seed_runs + n + loco_models + 1 + loco_folds
    return Estimate(
        n_main_candidates=n, seed_replicates=seed_replicates,
        main_seed_runs=main_seed_runs, chosen_main=n, loco_models=loco_models,
        c0=1, c0_profile_withheld=loco_folds, total_models=total,
    )


def record(model_id: str, wall_s: float, gpu_h: float, params: int, *, path: str | Path) -> None:
    """Write one run's totals, replacing its previous entry when a run is resumed.

    The caller owns the output directory. Existing ledger fields are preserved.
    """
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {"runs": {}}
    data["runs"][model_id] = {
        "wall_s": float(wall_s), "gpu_h": float(gpu_h), "params": int(params),
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def total(*, path: str | Path) -> dict:
    """Sum recorded runs; a ledger that has not been created has zero totals."""
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {"runs": {}}
    runs = data["runs"].values()
    return {
        "wall_s": sum(r["wall_s"] for r in runs),
        "gpu_h": sum(r["gpu_h"] for r in runs),
        "n_runs": len(data["runs"]),
    }


def plan_dict(candidates: Sequence[str], seed_replicates: int, loco_folds: int) -> dict:
    """Return the compute estimate in a JSON-compatible mapping."""
    return asdict(plan(candidates, seed_replicates, loco_folds))
