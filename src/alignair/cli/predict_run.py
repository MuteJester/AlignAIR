"""A ``predict`` run's two shared objects: how it reads its input and writes its rows (``RunIO``), and its record
accounting (``RunCounts``), which the streaming pass fills and the genotype pass corrects."""
from __future__ import annotations

from dataclasses import dataclass, field

ASSEMBLY_STATUSES = ("complete", "partial", "failed")


@dataclass(frozen=True)
class RunIO:
    """How a run reads its input and writes its rows: the input and its reader's options, the AIRR columns written
    (``--columns``) and the metadata's kept columns, the locus of a record that carries none, and the metadata join
    index (None without ``--metadata``)."""
    input_path: str
    chunk_size: int
    seq_column: str | None
    id_column: str | None
    columns: str | None
    extra_cols: list[str] | None
    out_locus: str
    meta_by_id: object = None


def outcome(record: dict) -> tuple[str | None, str | None]:
    """A row's assembly status and, for a partial or failed row, the reason the run metadata counts."""
    status = record.get("airr_assembly_status")
    if status == "partial":
        return status, record.get("airr_assembly_reason", "unknown")
    if status == "failed":
        return status, record.get("airr_assembly_error", "?").split(":")[0]
    return status, None


@dataclass
class RunCounts:
    """A run's record accounting: the reads accepted and rejected, the rows written and cropped, each assembly status
    and the reasons of the partial and failed rows (in the order first seen), and the rows in an orientation AIRR
    cannot represent."""
    accepted: int = 0
    rejected: int = 0
    cropped: int = 0
    complete: int = 0
    partial: int = 0
    failed: int = 0
    written: int = 0
    nonstandard_orientation: int = 0
    fail_reasons: dict[str, int] = field(default_factory=dict)
    partial_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def input(self) -> int:
        return self.accepted + self.rejected

    def add(self, record: dict) -> None:
        """Count one written row."""
        self._count(outcome(record), 1)
        if record.get("length_cropped"):
            self.cropped += 1
        if record.get("orientation_id") in (2, 3):   # complement/reverse-only: not AIRR-representable
            self.nonstandard_orientation += 1

    def move(self, before: list, after: list) -> None:
        """Move replaced rows' assembly outcomes (``before``) to their new ones (``after``)."""
        for sign, outcomes in ((-1, before), (1, after)):
            for row in outcomes:
                self._count(row, sign)

    def _count(self, row: tuple, sign: int) -> None:
        status, reason = row
        if status in ASSEMBLY_STATUSES:
            setattr(self, status, getattr(self, status) + sign)
        reasons = self.partial_reasons if status == "partial" else self.fail_reasons if status == "failed" else None
        if reasons is not None:
            reasons[reason] = reasons.get(reason, 0) + sign
            if not reasons[reason]:
                del reasons[reason]

    def record(self) -> dict:
        """The run metadata's ``counts``: the full record accounting."""
        return {"input_records": self.input, "accepted_records": self.accepted,
                "rejected_records": self.rejected, "cropped_records": self.cropped,
                "complete_assemblies": self.complete, "partial_assemblies": self.partial,
                "failed_assemblies": self.failed, "written_records": self.written,
                # complement/reverse-only reads: `sequence` is transformed and rev_comp can't express it
                # (see the `orientation` extension). A surprising rate is a useful QC signal.
                "nonstandard_orientation_records": self.nonstandard_orientation}
