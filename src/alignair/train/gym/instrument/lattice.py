"""FrozenLattice: the seeded, stratified, never-trained evaluation cells over the
TaskSpace. Includes explicit deployment-hard cells (heavy-SHM, full-length heavy-SHM,
junction-boundary) — the corners the coupled scalar ramp structurally never visits.
Fingerprinted so the instrument version is identifiable and comparable across runs."""
import hashlib
import json
from dataclasses import dataclass

from .task_space import TaskSpace


@dataclass(frozen=True)
class LatticeCell:
    name: str
    frac: dict          # axis -> fraction of range to FIX (others default mid)
    n: int


# fractions are of each axis's [lo,hi] range. full-length => crop_len frac 1.0 (=576).
# Two kinds of cell: COMPOSITE (clean + the hard corners we care about) and AXIS-ISOLATED
# (one axis stressed, every other axis at baseline) — the isolated cells give a clean
# per-axis competence signal that drives FactoredCurriculum pacing without the `clean`
# fallback confounding it.
_STANDARD = (
    # composite reference + hard corners
    LatticeCell("clean", {"mutation_rate": 0.0, "end_loss_5": 0.0, "end_loss_3": 0.0,
                          "indel_count": 0.0, "crop_len": 1.0, "orient_prob": 0.0}, 2000),
    LatticeCell("heavy_shm", {"mutation_rate": 0.85, "crop_len": 0.4}, 2000),
    LatticeCell("heavy_shm_fulllen", {"mutation_rate": 0.85, "crop_len": 1.0,
                                      "end_loss_5": 0.0, "end_loss_3": 0.0}, 2000),
    LatticeCell("junction_boundary", {"mutation_rate": 0.4, "indel_count": 0.5,
                                      "crop_len": 0.2}, 2000),
    LatticeCell("fragment", {"mutation_rate": 0.3, "crop_len": 0.0}, 2000),
    # axis-isolated cells (one axis stressed @ high frac, all others at baseline lo)
    LatticeCell("trim", {"end_loss_5": 0.7, "end_loss_3": 0.7, "crop_len": 1.0}, 2000),
    LatticeCell("indel", {"indel_count": 0.8, "crop_len": 1.0}, 2000),
    LatticeCell("ambiguous", {"ambiguous_count": 0.8, "crop_len": 1.0}, 2000),
    LatticeCell("seq_error", {"seq_error_rate": 0.9, "crop_len": 1.0}, 2000),
    LatticeCell("orient", {"orient_prob": 1.0, "crop_len": 1.0}, 2000),
)


class FrozenLattice:
    def __init__(self, task_space: TaskSpace, cells, seed: int):
        self.task_space = task_space
        self.cells = tuple(cells)
        self.seed = seed

    @classmethod
    def standard(cls, seed: int = 0):
        return cls(TaskSpace.deployment(), _STANDARD, seed)

    def cell_params(self, cell: LatticeCell) -> dict:
        # a deterministic difficulty POINT: only the cell's named axes are stressed,
        # every other axis sits at its easy baseline (so noise axes don't confound it).
        theta = self.task_space.sample(frac=cell.frac)
        return self.task_space.to_genairr_params(theta)

    def fingerprint(self) -> str:
        payload = {
            "seed": self.seed,
            "axes": [(a.name, a.lo, a.hi, a.kind) for a in self.task_space.axes],
            "cells": [(c.name, sorted(c.frac.items()), c.n) for c in self.cells],
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
