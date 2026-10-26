"""Stage-5 Task 5.3 — mechanical access barrier around the fresh-selection / loco-challenge boundary.

The owner's enforced invariants (this is the *gate*, not a convention):
  1. The fresh-selection loader CANNOT open challenge files (hard filesystem-boundary refusal).
  2. The challenge loader REFUSES to open unless a COMPLETE, hash-bound fold-threshold artifact already exists.
  3. That artifact must identify: fold checkpoint (provenance digest AND actual file SHA-256), selection
     manifest, excluded group/proxies, score directions, the finite grid, the mask hash, and the pre-frozen
     challenge-file digest.
  4. Challenge access writes an ATOMIC audit record BEFORE any read; then it hashes the opened bytes and — on
     mismatch with the pre-frozen challenge digest — writes an atomic audit FAILURE record and refuses.
  5. Any changed hash HARD-FAILS; there is NO silent recomputation.

The predeclared score directions, candidate-specific finite grids, per-fold gate, and selection-gate map are
frozen in ``decision``/``selection`` (before any fresh read) and hashed via ``threshold_grid_hash()``.
"""
from __future__ import annotations

from alignair.errors import AccessPolicyError, ArtifactIntegrityError, DataContractError

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from alignair.aa.guard.decision import COMPONENT_DIRECTIONS, COMPONENT_LEVELS, CandidateSpec, candidate_grid
from alignair.aa.guard.dual_eval import ModeScores, assert_binding_is_blind
from alignair.aa.guard.loco import LOCO_MASK_HASH, masked_profiles_for
from alignair.aa.guard.selection import SELECTION_GATES, select_thresholds

FOLD_GATE = {"igk_igl": "gate_5pct", "tcr": "gate_5pct", "vhh_nonhuman": "gate_10pct", "igsf": "gate_10pct"}


def _sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def threshold_grid_hash() -> str:
    """Digest of the FULL frozen decision spec: every candidate's finite grid + score directions + component
    levels + the per-group selection gates + the per-fold gate (bundle slot ``threshold_grid``)."""
    return _sha_json({
        "candidate_grids": {cid: [dict(g) for g in candidate_grid(cid)] for cid in ("C0", "C1", "C2", "C3", "C4")},
        "score_directions": COMPONENT_DIRECTIONS, "component_levels": {k: list(v) for k, v in COMPONENT_LEVELS.items()},
        "selection_gates": SELECTION_GATES, "fold_gate": FOLD_GATE})


class AccessViolation(AccessPolicyError):
    """A loader was asked to open a file outside its permitted role (e.g. selection loader → challenge file)."""
    code = "ALIGNAIR_ACCESS_VIOLATION"


class ThresholdArtifactMissing(DataContractError):
    """Challenge access attempted with no complete, hash-bound fold-threshold artifact present."""
    code = "ALIGNAIR_THRESHOLD_ARTIFACT_MISSING"


class HashMismatch(ArtifactIntegrityError):
    """A bound identity (checkpoint bytes, mask, grid, bundle, selection manifest, or challenge file) drifted."""
    code = "ALIGNAIR_IDENTITY_MISMATCH"


class NoAdmissibleThreshold(DataContractError):
    """No grid point satisfied every applicable selection gate — a STOP, never a relaxation."""
    code = "ALIGNAIR_NO_ADMISSIBLE_THRESHOLD"


# ---- fold-threshold artifact -------------------------------------------------------------------------
@dataclass(frozen=True)
class FoldThresholdArtifact:
    fold: str
    candidate: str
    fold_checkpoint_sha256: str            # combined model identity (== bundle "model" slot)
    checkpoint_provenance_digest: str      # (3) metadata identity
    checkpoint_file_sha256: str            # (3) actual weight-file bytes identity
    selection_manifest_sha256: str         # (3) selection manifest it was frozen on
    challenge_file_sha256: str             # (3) pre-frozen challenge-file digest (referenced, not inspected)
    excluded_group: str                    # (3) held-out cohort group
    excluded_proxies: tuple                # (3) masked profile groups (proxies)
    score_directions: dict                 # (3) frozen score directions
    grid: tuple                            # (3) the candidate's finite grid
    mask_hash: str                         # (3) LOCO_MASK_HASH
    gate_kind: str                         # the held-out fold's own FAR gate
    bundle_sha256: str                     # the pre-unlock bundle this threshold was bound to
    chosen_grid_point: dict                # the frozen conjunctive operating point
    per_group: dict                        # full per-group k/n/estimate/upper/ceiling/pass table (no pooling)
    required_groups: tuple                 # the groups that had to pass on the selection subset
    mock: bool = False                     # True for the synthetic one-shot (never a production operating point)
    marker: str = ""

    def is_complete(self) -> bool:
        return bool(self.fold and self.candidate and self.fold_checkpoint_sha256
                    and self.checkpoint_provenance_digest and self.checkpoint_file_sha256
                    and self.selection_manifest_sha256 and self.challenge_file_sha256 and self.excluded_group
                    and self.score_directions and self.grid and self.mask_hash and self.gate_kind
                    and self.bundle_sha256 and self.chosen_grid_point and self.per_group)

    def grid_digest(self) -> str:
        return threshold_grid_hash()

    def digest(self) -> str:
        return _sha_json(self.to_dict())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["grid"] = [dict(g) for g in self.grid]
        d["excluded_proxies"] = list(self.excluded_proxies)
        d["required_groups"] = list(self.required_groups)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FoldThresholdArtifact":
        return cls(fold=d["fold"], candidate=d["candidate"], fold_checkpoint_sha256=d["fold_checkpoint_sha256"],
                   checkpoint_provenance_digest=d["checkpoint_provenance_digest"],
                   checkpoint_file_sha256=d["checkpoint_file_sha256"],
                   selection_manifest_sha256=d["selection_manifest_sha256"],
                   challenge_file_sha256=d["challenge_file_sha256"], excluded_group=d["excluded_group"],
                   excluded_proxies=tuple(d["excluded_proxies"]), score_directions=d["score_directions"],
                   grid=tuple(dict(g) for g in d["grid"]), mask_hash=d["mask_hash"], gate_kind=d["gate_kind"],
                   bundle_sha256=d["bundle_sha256"], chosen_grid_point=d["chosen_grid_point"],
                   per_group=d["per_group"], required_groups=tuple(d["required_groups"]),
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
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def freeze_fold_threshold(fold, candidate, blind_selection: ModeScores, required_groups, *,
                          model_identity_hash, checkpoint_provenance_digest, checkpoint_file_sha256,
                          selection_manifest_sha256, challenge_file_sha256, frozen_bundle, out_path,
                          grid=None, mock=False, marker="") -> FoldThresholdArtifact:
    """Select ONE conjunctive operating point on the BLIND selection scores (blind is the only mode that may
    bind), with all per-group failure counts DERIVED INTERNALLY (no caller counts), and bind it to the frozen
    bundle. Raises ``NoAdmissibleThreshold`` (STOP) if no grid point passes every required gate."""
    assert_binding_is_blind(blind_selection)                          # reference_covered can never bind
    spec = CandidateSpec.of(candidate)
    used_grid = tuple(grid) if grid is not None else candidate_grid(candidate)
    sel = select_thresholds(spec, blind_selection.scores, required_groups, grid=used_grid)
    if sel is None:
        raise NoAdmissibleThreshold(f"no admissible operating point for {candidate} fold {fold!r} — STOP")
    art = FoldThresholdArtifact(
        fold=fold, candidate=candidate, fold_checkpoint_sha256=model_identity_hash,
        checkpoint_provenance_digest=checkpoint_provenance_digest, checkpoint_file_sha256=checkpoint_file_sha256,
        selection_manifest_sha256=selection_manifest_sha256, challenge_file_sha256=challenge_file_sha256,
        excluded_group=fold, excluded_proxies=tuple(sorted(masked_profiles_for(fold))),
        score_directions=COMPONENT_DIRECTIONS, grid=used_grid, mask_hash=LOCO_MASK_HASH, gate_kind=FOLD_GATE[fold],
        bundle_sha256=frozen_bundle.bundle_sha256(), chosen_grid_point=sel.grid_point, per_group=sel.per_group,
        required_groups=tuple(required_groups), mock=mock, marker=marker)
    _atomic_write_json(out_path, art.to_dict())
    return art


def verify_challenge_access(artifact: FoldThresholdArtifact, frozen_bundle, *,
                            expected_selection_manifest_sha256, expected_excluded,
                            expected_challenge_digest) -> None:
    """Hard-fail (no recompute) if the artifact is incomplete or any bound identity drifted from the frozen
    bundle / registry. Invariant (5): a changed hash stops the pipeline, it never silently re-derives."""
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
        "challenge_digest_ref": (artifact.challenge_file_sha256, expected_challenge_digest),
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
    """(2),(4),(5): opens challenge files only after a complete, hash-verified fold-threshold artifact exists;
    writes an atomic audit record BEFORE reading; then verifies the opened bytes against the pre-frozen
    challenge digest, writing an atomic audit FAILURE record and refusing on mismatch."""
    challenge_root: Path
    selection_root: Path
    threshold_artifact_path: Path
    audit_path: Path
    frozen_bundle: object
    expected_selection_manifest_sha256: str
    expected_excluded: tuple
    expected_challenge_digest: str
    marker: str = ""
    _opened: list = field(default_factory=list)

    def _load_artifact(self) -> FoldThresholdArtifact:
        p = Path(self.threshold_artifact_path)
        if not p.exists():
            raise ThresholdArtifactMissing(f"no fold-threshold artifact at {p} — challenge access refused")
        return FoldThresholdArtifact.from_dict(json.loads(p.read_text()))

    def _atomic_audit(self, record: dict) -> None:
        """Append one audit record, whole-file temp+replace (atomic)."""
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
                                expected_excluded=self.expected_excluded,
                                expected_challenge_digest=self.expected_challenge_digest)
        self._atomic_audit({"event": "challenge_unlock", "fold": artifact.fold, "candidate": artifact.candidate,
                            "challenge_path": str(p), "artifact_digest": artifact.digest(),
                            "bundle_sha256": artifact.bundle_sha256, "ts_unix": time.time(),
                            "marker": self.marker})                   # (4) audit BEFORE reading
        data = p.read_bytes()
        got = hashlib.sha256(data).hexdigest()
        if got != self.expected_challenge_digest:                    # (7) opened bytes must match pre-frozen digest
            self._atomic_audit({"event": "challenge_digest_mismatch", "fold": artifact.fold,
                                "challenge_path": str(p), "expected": self.expected_challenge_digest,
                                "got": got, "ts_unix": time.time(), "failure": True})
            raise HashMismatch(f"challenge file digest mismatch: expected {self.expected_challenge_digest[:12]}, "
                               f"got {got[:12]} — challenge bytes not the pre-frozen file")
        self._opened.append(str(p))
        return data
