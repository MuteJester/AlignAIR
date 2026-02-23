"""Central store of all pipeline intermediates."""
from __future__ import annotations

from typing import Any, Dict, FrozenSet

from AlignAIR.Pipeline.Stage.protocol import StageContext


# Slots that are allowed to be overwritten by later stages
OVERWRITABLE_SLOTS = frozenset({
    "processed_predictions",  # genotype adjustment, segment correction
    "selected_allele_calls",  # translate names
    "sequences",              # batch inference overwrites with orientation-fixed
    "corrected_segments",
    "likelihoods_of_selected_alleles",
})


class SlotStore:
    """Central store of all pipeline intermediates.

    Slots are write-once by default. Specific slots (like processed_predictions)
    can be overwritten by stages that transform them in-place (e.g. genotype adjustment).
    """

    def __init__(self):
        self._slots: Dict[str, Any] = {}

    def set(self, key: str, value: Any, allow_overwrite: bool = False) -> None:
        if key in self._slots and not allow_overwrite and key not in OVERWRITABLE_SLOTS:
            raise RuntimeError(
                f"Slot '{key}' already set and is not overwritable. "
                f"Available overwritable slots: {OVERWRITABLE_SLOTS}"
            )
        self._slots[key] = value

    def get(self, key: str) -> Any:
        if key not in self._slots:
            raise KeyError(
                f"Slot '{key}' not found. Available: {sorted(self._slots.keys())}"
            )
        return self._slots[key]

    def has(self, key: str) -> bool:
        return key in self._slots

    def keys(self) -> FrozenSet[str]:
        return frozenset(self._slots.keys())

    def release(self, key: str) -> None:
        """Explicitly free a slot's data for memory reclamation."""
        self._slots.pop(key, None)

    def build_context(self, required_keys: FrozenSet[str]) -> StageContext:
        """Build a read-only StageContext with only the requested slots."""
        # Always include 'config' if available
        keys_with_config = required_keys | ({"config"} if "config" in self._slots else set())
        subset = {k: self._slots[k] for k in keys_with_config if k in self._slots}
        return StageContext(subset, required_keys)
