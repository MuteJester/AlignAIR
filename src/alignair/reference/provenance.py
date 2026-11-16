"""Where a reference's alleles came from: one ``SourceProvenance`` per source (a GenAIRR dataconfig, an IMGT or OGRDB
file, a user's FASTA), with what identifies it (``locator``, ``sha256``, ``release``), when it was retrieved, and its
licence and attribution, which a model card and an export carry."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceProvenance:
    """One source of a reference's alleles (see the module docstring)."""
    kind: str
    locator: str | None = None
    sha256: str | None = None
    release: str | None = None
    retrieved_at: str | None = None
    licence: str | None = None
    attribution: str | None = None

    def to_dict(self) -> dict:
        """The fields as a plain dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SourceProvenance":
        """From ``to_dict``'s form; unknown keys are ignored."""
        names = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in dict(data).items() if key in names})
