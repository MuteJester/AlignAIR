"""Deterministic allocation of unique output IDs without changing source join keys."""
from ..errors import DataContractError


class UniqueSequenceIds:
    """Allocate in source order; every emitted name is reserved, including generated suffixes.

    One allocator must span all chunks of an input. It retains IDs, not sequence payloads.
    A later real header may already be occupied by a generated name and is then disambiguated.
    Callers retain the original source ID separately for metadata lookup.
    """

    def __init__(self):
        self._used: set[str] = set()
        self._next_suffix: dict[str, int] = {}

    def assign(self, base: str) -> str:
        if not isinstance(base, str) or not base:
            raise DataContractError("sequence ID base must be a nonempty string")
        candidate = base
        suffix = self._next_suffix.get(base, 1)
        while candidate in self._used:
            candidate = f"{base}_dup{suffix}"
            suffix += 1
        self._used.add(candidate)
        self._next_suffix[base] = suffix
        return candidate
