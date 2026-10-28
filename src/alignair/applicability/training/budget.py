"""Compute estimates and per-run accounting for guard training studies."""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path

from alignair.errors import ArtifactIntegrityError, InputError, NumericalError, OperationIOError, error_context
from alignair.io.atomic import atomic_write_json


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


def _validate_entry(entry, error_type):
    if not isinstance(entry, dict):
        raise error_type("Accounting run must be a mapping.")
    for field in ("wall_s", "gpu_h"):
        value = entry.get(field)
        try:
            valid = isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise error_type("Accounting durations must be finite nonnegative numbers.", context={"field": field})
    value = entry.get("params")
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise error_type("Accounting parameter count must be a nonnegative integer.", context={"field": "params"})


def _read_ledger(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON number exceeds finite numeric range")
        return number
    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant, parse_float=finite_float)
    except FileNotFoundError:
        return {"runs": {}}
    except ValueError as exc:
        raise ArtifactIntegrityError("Accounting ledger is not finite UTF-8 JSON.") from exc
    except OSError as exc:
        raise OperationIOError("Cannot read accounting ledger.") from exc
    if not isinstance(data, dict) or not isinstance(data.get("runs"), dict):
        raise ArtifactIntegrityError("Accounting ledger requires a runs mapping.")
    for model_id, entry in data["runs"].items():
        with error_context(model_id=model_id):
            if not model_id:
                raise ArtifactIntegrityError("Accounting run ID must be nonempty.")
            _validate_entry(entry, ArtifactIntegrityError)
    return data


def record(model_id: str, wall_s: float, gpu_h: float, params: int, *, path: str | Path) -> None:
    """Write one run's totals, replacing its previous entry when a run is resumed.

    Existing ledger fields and other runs are preserved. Updates require one serialized writer;
    per-file atomic publication does not lock a read-modify-write transaction.
    """
    path = Path(path)
    with error_context(stage="guard_training_budget", path=str(path), model_id=model_id):
        if not isinstance(model_id, str) or not model_id:
            raise InputError("Accounting run ID must be nonempty text.", context={"field": "model_id"})
        entry = {"wall_s": wall_s, "gpu_h": gpu_h, "params": params}
        _validate_entry(entry, InputError)
        data = _read_ledger(path)
        data["runs"][model_id] = {"wall_s": float(wall_s), "gpu_h": float(gpu_h), "params": int(params)}
        atomic_write_json(path, data, stage="guard_training_budget")


def total(*, path: str | Path) -> dict:
    """Sum recorded runs; a ledger that has not been created has zero totals."""
    path = Path(path)
    with error_context(stage="guard_training_budget", path=str(path)):
        data = _read_ledger(path)
        runs = data["runs"].values()
        result = {
            "wall_s": sum(r["wall_s"] for r in runs),
            "gpu_h": sum(r["gpu_h"] for r in runs),
            "n_runs": len(data["runs"]),
        }
        try:
            finite = math.isfinite(result["wall_s"]) and math.isfinite(result["gpu_h"])
        except OverflowError:
            finite = False
        if not finite:
            raise NumericalError("Accounting totals exceed finite numeric range.")
        return result


def plan_dict(candidates: Sequence[str], seed_replicates: int, loco_folds: int) -> dict:
    """Return the compute estimate in a JSON-compatible mapping."""
    return asdict(plan(candidates, seed_replicates, loco_folds))
