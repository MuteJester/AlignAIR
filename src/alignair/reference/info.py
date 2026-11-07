"""What a source tells about one allele: its functionality, species, source label, and how its anchor was found.

``AnchorEvidence`` records the method (``supplied``, ``known_allele`` or ``nearest_known_allele``, set by AlignAIR, or
GenAIRR's resolver's own method name: ``imgt_gapped_104``, ``unique_j_fwgxg``, ``unresolved``) and the outcome
(``ANCHOR_STATUSES``); ``ANCHORED`` are the statuses that mean the allele has an anchor. Both types are frozen and
serialize to plain dicts whose unknown keys a reader ignores."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

ANCHOR_STATUSES = ("supplied", "inferred", "verified", "ambiguous", "unresolved", "invalid")
ANCHORED = ("supplied", "inferred", "verified")
FUNCTIONALITIES = ("F", "ORF", "P")


def _known(cls, data: dict) -> dict:
    names = {field.name for field in dataclasses.fields(cls)}
    return {key: value for key, value in dict(data).items() if key in names}


@dataclass(frozen=True)
class AnchorEvidence:
    method: str
    status: str
    detail: str | None = None
    source: str | None = None

    def __post_init__(self):
        if self.status not in ANCHOR_STATUSES:
            raise ValueError(f"an anchor's status is one of {', '.join(ANCHOR_STATUSES)}, got {self.status!r}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AnchorEvidence":
        return cls(**_known(cls, data))


@dataclass(frozen=True)
class AlleleInfo:
    functional: str | None = None
    species: str | None = None
    source_label: str | None = None
    anchor_evidence: AnchorEvidence | None = None

    def __post_init__(self):
        if self.functional is not None and self.functional not in FUNCTIONALITIES:
            raise ValueError(f"an allele's functionality is one of {', '.join(FUNCTIONALITIES)} or None, "
                             f"got {self.functional!r}")

    def to_dict(self) -> dict:
        """The fields that are set; the evidence as its own dict."""
        out = {key: value for key, value in (("functional", self.functional), ("species", self.species),
                                             ("source_label", self.source_label)) if value is not None}
        if self.anchor_evidence is not None:
            out["anchor_evidence"] = self.anchor_evidence.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "AlleleInfo":
        """Read leniently, as a newer v1 file may carry values this build does not know: an unknown functionality
        reads as None and evidence with an unknown status is dropped."""
        fields = _known(cls, data)
        if fields.get("functional") not in (None, *FUNCTIONALITIES):
            fields["functional"] = None
        evidence = fields.get("anchor_evidence")
        if evidence is not None:
            fields["anchor_evidence"] = (AnchorEvidence.from_dict(evidence)
                                         if dict(evidence).get("status") in ANCHOR_STATUSES else None)
        return cls(**fields)
