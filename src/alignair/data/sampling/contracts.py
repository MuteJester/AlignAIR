"""Source-independent sample preparation and selection interfaces."""
from collections.abc import Mapping, Sequence
from typing import NamedTuple, Protocol, TypeVar

RecordType = TypeVar("RecordType", contravariant=True)


class SampleResult(NamedTuple):
    sample: Mapping | None
    reason: str | None = None


class SampleBuilder(Protocol[RecordType]):
    def __call__(self, record: RecordType) -> SampleResult: ...


class SampleSelector(Protocol):
    def __call__(self, sample: Mapping, selected: Sequence[Mapping]) -> bool: ...
