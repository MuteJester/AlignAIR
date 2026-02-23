"""Stage protocol — the base contract for all pipeline stages."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, FrozenSet

from AlignAIR.Pipeline.Errors.exceptions import StageContractError


class StageContext:
    """Read-only, typed view of pipeline slots for a stage.

    The runner constructs this with only the slots declared in
    the stage's `reads` set. Stages cannot access undeclared slots
    (except 'config', which is always available).
    """

    def __init__(self, slots: Dict[str, Any], available_keys: FrozenSet[str]):
        self._slots = slots
        self._available = available_keys | {"config"}

    def __getitem__(self, key: str) -> Any:
        if key not in self._available:
            raise KeyError(
                f"Slot '{key}' not declared in stage's reads set. "
                f"Available: {sorted(self._available)}"
            )
        if key not in self._slots:
            raise KeyError(
                f"Slot '{key}' is declared but not yet populated. "
                f"Check pipeline stage ordering."
            )
        return self._slots[key]

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._available:
            return default
        return self._slots.get(key, default)

    @property
    def config(self):
        """Shortcut — config is always available."""
        return self._slots["config"]


class Stage(ABC):
    """Base protocol for all pipeline stages.

    Stages are stateless processors. They read named slots from the
    pipeline context, produce new named slots, and return them.

    Subclasses MUST declare:
        reads:  frozenset of slot names this stage requires
        writes: frozenset of slot names this stage produces

    The pipeline runner uses these declarations to:
        1. Validate the pipeline DAG before execution
        2. Pass only the required slots to each stage
        3. Verify outputs match the declared writes
    """
    reads: FrozenSet[str] = frozenset()
    writes: FrozenSet[str] = frozenset()

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, context: StageContext) -> Dict[str, Any]:
        """Execute this stage.

        Args:
            context: Read-only view of the pipeline slots declared in `reads`.

        Returns:
            Dict mapping slot names (from `writes`) to their values.
            All declared writes MUST be present in the return dict.
        """
        ...

    def validate_inputs(self, context: StageContext) -> None:
        """Optional pre-execution validation. Override in subclass."""
        pass

    def validate_outputs(self, outputs: Dict[str, Any]) -> None:
        """Verify all declared writes are present in outputs."""
        if not outputs:
            return
        missing = self.writes - set(outputs.keys())
        if missing:
            raise StageContractError(
                f"Stage '{self.name}' failed to produce declared outputs: {missing}"
            )


class ConditionalStage(Stage):
    """A stage that may be skipped based on runtime conditions.

    Subclasses implement `should_run()`. If it returns False,
    the runner skips this stage and no outputs are produced.
    """

    @abstractmethod
    def should_run(self, context: StageContext) -> bool:
        """Return True if this stage should execute."""
        ...

    def run(self, context: StageContext) -> Dict[str, Any]:
        if not self.should_run(context):
            return {}
        return self._run(context)

    @abstractmethod
    def _run(self, context: StageContext) -> Dict[str, Any]:
        """Actual stage logic when condition is met."""
        ...
