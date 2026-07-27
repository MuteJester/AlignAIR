"""Stage-5 Task 5.3 — mechanical access barrier around the fresh-selection / loco-challenge boundary.

The owner's five enforced invariants (this is the *gate*, not a convention):
  1. The fresh-selection loader CANNOT open challenge files (hard filesystem-boundary refusal).
  2. The challenge loader REFUSES to open unless a COMPLETE, hash-bound fold-threshold artifact already exists.
  3. That artifact must identify: fold checkpoint, selection manifest, excluded group/proxies, score
     directions, the finite grid, and the mask hash.
  4. Challenge access writes an ATOMIC audit record BEFORE any scoring/read of challenge data.
  5. Any changed hash HARD-FAILS; there is NO silent recomputation.

The predeclared score directions, finite grid, and per-fold gate are frozen HERE (before any fresh read) and
hashed into the pre-unlock bundle via ``threshold_grid_hash()``.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from alignair.aa.guard.dual_eval import ModeScores, assert_binding_is_blind
from alignair.aa.guard.loco import LOCO_MASK_HASH, masked_profiles_for
from alignair.aa.guard.thresholds import select_thresholds

# ---- frozen decision spec (predeclared BEFORE any fresh data is seen) --------------------------------
SCORE_DIRECTIONS = {
    "applicability_logit": "higher_is_more_supported",
    "energy": "lower_is_more_in_domain",
    "c0_margin": "higher_is_more_human_igh",
}
# Finite operating-point grid: candidate applicability-logit cutoffs, in predeclared order.
PREDECLARED_GRID = tuple({"applicability_logit_cutoff": round(c, 3)}
                         for c in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0))
FOLD_GATE = {"igk_igl": "gate_5pct", "tcr": "gate_5pct", "vhh_nonhuman": "gate_10pct", "igsf": "gate_10pct"}


def _sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def threshold_grid_hash() -> str:
    """Digest of the frozen grid + score directions + per-fold gate map (bundle slot ``threshold_grid``)."""
    return _sha_json({"grid": [dict(g) for g in PREDECLARED_GRID],
                      "score_directions": SCORE_DIRECTIONS, "fold_gate": FOLD_GATE})


class AccessViolation(RuntimeError):
    """A loader was asked to open a file outside its permitted role (e.g. selection loader → challenge file)."""


class ThresholdArtifactMissing(RuntimeError):
    """Challenge access attempted with no complete, hash-bound fold-threshold artifact present."""


class HashMismatch(RuntimeError):
    """A bound identity in the fold-threshold artifact no longer matches the frozen bundle — hard stop."""


# ---- fold-threshold artifact -------------------------------------------------------------------------
@dataclass(frozen=True)
class FoldThresholdArtifact:
    fold: str
    fold_checkpoint_sha256: str        # (3) identifies the fold checkpoint
    selection_manifest_sha256: str     # (3) identifies the selection manifest it was frozen on
    excluded_group: str                # (3) held-out cohort group
    excluded_proxies: tuple            # (3) masked profile groups (proxies)
    score_directions: dict             # (3) frozen score directions
    grid: tuple                        # (3) the finite operating-point grid
    mask_hash: str                     # (3) LOCO_MASK_HASH
    gate_kind: str
    bundle_sha256: str                 # the pre-unlock bundle this threshold was bound to
    chosen_grid_point: dict            # the frozen operating point (selected on the blind selection subset)
    far_upper: float
    frr_upper: float
    n_neg: int
    n_pos: int
    mock: bool = False                 # True for the synthetic one-shot (never a production operating point)
    marker: str = ""                   # provenance marker (audit note)

    def is_complete(self) -> bool:
        """(2)+(3): all identity fields present. A missing identity = an incomplete artifact = refuse."""
        return bool(self.fold and self.fold_checkpoint_sha256 and self.selection_manifest_sha256
                    and self.excluded_group and self.score_directions and self.grid
                    and self.mask_hash and self.gate_kind and self.bundle_sha256 and self.chosen_grid_point)

    def grid_digest(self) -> str:
        return _sha_json({"grid": [dict(g) for g in self.grid], "score_directions": self.score_directions,
                          "fold_gate": FOLD_GATE})

    def digest(self) -> str:
        return _sha_json(self.to_dict())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["grid"] = [dict(g) for g in self.grid]
        d["excluded_proxies"] = list(self.excluded_proxies)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FoldThresholdArtifact":
        return cls(fold=d["fold"], fold_checkpoint_sha256=d["fold_checkpoint_sha256"],
                   selection_manifest_sha256=d["selection_manifest_sha256"], excluded_group=d["excluded_group"],
                   excluded_proxies=tuple(d["excluded_proxies"]), score_directions=d["score_directions"],
                   grid=tuple(dict(g) for g in d["grid"]), mask_hash=d["mask_hash"], gate_kind=d["gate_kind"],
                   bundle_sha256=d["bundle_sha256"], chosen_grid_point=d["chosen_grid_point"],
                   far_upper=d["far_upper"], frr_upper=d["frr_upper"], n_neg=d["n_neg"], n_pos=d["n_pos"],
                   mock=d.get("mock", False), marker=d.get("marker", ""))


def _atomic_write_json(path, obj) -> None:
    """Write JSON via a same-dir temp file + os.replace (atomic on POSIX)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, sort_keys=True, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)                       # atomic swap
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def freeze_fold_threshold(fold, blind_selection: ModeScores, *, fold_checkpoint_sha256,
                          selection_manifest_sha256, frozen_bundle, guardband_grid, out_path,
                          mock=False, marker="") -> FoldThresholdArtifact:
    """Select ONE operating point from the frozen grid on the BLIND selection scores (blind is the only mode
    that may bind), bind it to the frozen bundle, and write the artifact atomically.

    ``guardband_grid``: iterable of (grid_point, far_k, far_n, frr_k, frr_n) — the exact-bound inputs from the
    selection subset. Refuses if the binding mode is not blind, or if no grid point passes the guardband."""
    assert_binding_is_blind(blind_selection)                          # reference_covered can never bind
    gate_kind = FOLD_GATE[fold]
    chosen = select_thresholds(fold, blind_selection.scores, guardband_grid, gate_kind=gate_kind)
    if chosen is None:
        raise ValueError(f"no admissible operating point for fold {fold!r} under {gate_kind} guardband")
    art = FoldThresholdArtifact(
        fold=fold, fold_checkpoint_sha256=fold_checkpoint_sha256,
        selection_manifest_sha256=selection_manifest_sha256, excluded_group=fold,
        excluded_proxies=tuple(sorted(masked_profiles_for(fold))), score_directions=SCORE_DIRECTIONS,
        grid=PREDECLARED_GRID, mask_hash=LOCO_MASK_HASH, gate_kind=gate_kind,
        bundle_sha256=frozen_bundle.bundle_sha256(), chosen_grid_point=chosen.grid_point,
        far_upper=chosen.far_upper, frr_upper=chosen.frr_upper, n_neg=chosen.n_neg, n_pos=chosen.n_pos,
        mock=mock, marker=marker)
    _atomic_write_json(out_path, art.to_dict())
    return art


def verify_challenge_access(artifact: FoldThresholdArtifact, frozen_bundle, *,
                            expected_selection_manifest_sha256, expected_excluded) -> None:
    """Hard-fail (no recompute) if the artifact is incomplete or any bound identity drifted from the frozen
    bundle. This is invariant (5): a changed hash stops the pipeline, it never silently re-derives."""
    if not artifact.is_complete():
        raise ThresholdArtifactMissing(f"incomplete fold-threshold artifact for fold {artifact.fold!r}")
    b = frozen_bundle.hashes
    checks = {
        "model/fold_checkpoint": (artifact.fold_checkpoint_sha256, b["model"]),
        "mask_graph": (artifact.mask_hash, b["mask_graph"]),
        "mask_graph==LOCO": (artifact.mask_hash, LOCO_MASK_HASH),
        "threshold_grid": (artifact.grid_digest(), b["threshold_grid"]),
        "bundle": (artifact.bundle_sha256, frozen_bundle.bundle_sha256()),
        "selection_manifest": (artifact.selection_manifest_sha256, expected_selection_manifest_sha256),
    }
    drift = [name for name, (got, exp) in checks.items() if got != exp]
    if (artifact.excluded_group, tuple(artifact.excluded_proxies)) != tuple(expected_excluded):
        drift.append("excluded_group/proxies")
    if drift:
        raise HashMismatch(f"fold-threshold artifact drift (no silent recompute): {drift}")


# ---- role-scoped loaders -----------------------------------------------------------------------------
def _resolved_under(path, root) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


class SelectionLoader:
    """(1) Reads ONLY fresh-selection files under ``selection_root``; structurally cannot reach ``challenge_root``."""

    def __init__(self, selection_root, challenge_root):
        self.selection_root = Path(selection_root)
        self.challenge_root = Path(challenge_root)

    def load(self, path) -> bytes:
        p = Path(path)
        if _resolved_under(p, self.challenge_root):
            raise AccessViolation(f"fresh-selection loader cannot access challenge file: {p}")
        if not _resolved_under(p, self.selection_root):
            raise AccessViolation(f"path {p} is not under the fresh-selection root {self.selection_root}")
        return p.read_bytes()


@dataclass
class ChallengeLoader:
    """(2),(4),(5): opens challenge files only after a complete, hash-verified fold-threshold artifact exists,
    and writes an atomic audit record BEFORE returning any challenge bytes."""
    challenge_root: Path
    selection_root: Path
    threshold_artifact_path: Path
    audit_path: Path
    frozen_bundle: object
    expected_selection_manifest_sha256: str
    expected_excluded: tuple
    marker: str = ""
    _opened: list = field(default_factory=list)

    def _load_artifact(self) -> FoldThresholdArtifact:
        p = Path(self.threshold_artifact_path)
        if not p.exists():
            raise ThresholdArtifactMissing(f"no fold-threshold artifact at {p} — challenge access refused")
        return FoldThresholdArtifact.from_dict(json.loads(p.read_text()))

    def _atomic_audit(self, record: dict) -> None:
        """Append one audit record, whole-file temp+replace (atomic). Written BEFORE any challenge read."""
        path = Path(self.audit_path)
        existing = []
        if path.exists():
            existing = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        existing.append(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write("\n".join(json.dumps(r, sort_keys=True) for r in existing) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def open(self, path) -> bytes:
        p = Path(path)
        if _resolved_under(p, self.selection_root):
            raise AccessViolation(f"challenge loader cannot read a selection file: {p}")
        if not _resolved_under(p, self.challenge_root):
            raise AccessViolation(f"path {p} is not under the challenge root {self.challenge_root}")
        artifact = self._load_artifact()                             # (2) refuse if missing/incomplete
        verify_challenge_access(artifact, self.frozen_bundle,        # (5) hard-fail on any drift, no recompute
                                expected_selection_manifest_sha256=self.expected_selection_manifest_sha256,
                                expected_excluded=self.expected_excluded)
        self._atomic_audit({"event": "challenge_unlock", "fold": artifact.fold, "challenge_path": str(p),
                            "artifact_digest": artifact.digest(), "bundle_sha256": artifact.bundle_sha256,
                            "ts_unix": time.time(), "marker": self.marker})   # (4) audit BEFORE scoring
        data = p.read_bytes()                                        # only now is challenge data exposed
        self._opened.append(str(p))
        return data
