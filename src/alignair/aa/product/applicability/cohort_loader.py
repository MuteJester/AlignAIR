"""Sealed loader for the frozen Gate-3 cohort.

Operationally seals the locked test: Tasks 4-10 (Stage-1 rules, Stage-2 membership, threshold selection)
can reach ONLY fit/validation via ``dev()``. The test set lives in a separate physical file and can be
obtained only through ``unlock_locked_test(reason)`` — a loud, audited, one-shot action reserved for the
Task-11 final evaluation. So "locked" is a procedural guarantee, not just a label a developer can peek past.

The freeze writes two partitions into the git-ignored store: ``dev_cohort.jsonl`` (fit+val) and
``locked_test.jsonl`` (test). ``dev()`` never opens the test file.
"""
from __future__ import annotations

from alignair.errors import AccessPolicyError

import json
import os

STORE = ".private/aa-gate3-hydration"
_DEV = "dev_cohort.jsonl"
_LOCKED = "locked_test.jsonl"
_AUDIT = "locked_test_unlock_audit.jsonl"


class LockedTestError(AccessPolicyError):
    """Raised on any attempt to reach the locked test set outside the sanctioned one-shot unlock."""
    code = "ALIGNAIR_LOCKED_TEST_ACCESS"


class GateCohort:
    def __init__(self, store: str = STORE):
        self.store = store

    def _read(self, name: str) -> list:
        path = os.path.join(self.store, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found — run the Task-3 freeze first")
        return [json.loads(l) for l in open(path)]

    def dev(self, split: str | None = None) -> list:
        """fit+val records for Tasks 4-10. ``split`` in {None, 'fit', 'val'}; 'test' is forbidden here
        and raises — the test set is not in this file."""
        if split == "test":
            raise LockedTestError(
                "the locked test is sealed; call unlock_locked_test(reason) in Task 11 only")
        if split not in (None, "fit", "val"):
            raise ValueError(f"split must be None/'fit'/'val', got {split!r}")
        recs = self._read(_DEV)
        return [r for r in recs if split is None or r["split"] == split]

    def unlock_locked_test(self, reason: str, *, allow_reunlock: bool = False) -> list:
        """ONE-SHOT, audited unlock of the locked test for the Task-11 final evaluation. Writes an audit
        record and returns the sealed test records. Raises if already unlocked (unless allow_reunlock),
        so an accidental second peek is a hard error, not a silent read."""
        if not reason or len(reason.strip()) < 10:
            raise ValueError("unlock requires a substantive reason (>= 10 chars)")
        audit_path = os.path.join(self.store, _AUDIT)
        already = os.path.exists(audit_path) and os.path.getsize(audit_path) > 0
        if already and not allow_reunlock:
            raise LockedTestError(
                f"locked test already unlocked (see {audit_path}); pass allow_reunlock=True to override")
        with open(audit_path, "a") as fh:
            fh.write(json.dumps({"event": "locked_test_unlocked", "reason": reason.strip()}) + "\n")
        return self._read(_LOCKED)
