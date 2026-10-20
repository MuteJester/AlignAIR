"""Cases: one simulated read as presented to a tool, with its GenAIRR record and labels.

A case is a JSON object with ``case_id``, ``suite``, ``stratum``, ``orientation`` (0 forward, 1 reverse
complement), ``sequence`` (the presented read, uppercase; the reverse complement of the record sequence when the
orientation is 1), ``record`` (the forward GenAIRR record, verbatim), ``labels`` (slice dimensions) and
``groups`` (grouping keys that are not sliced, such as ``subject``, ``clone`` and ``pair``).
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TypeVar

from alignair_assay.core.jsonl import read_jsonl, write_jsonl
from alignair_assay.core.seqops import revcomp

T = TypeVar("T")


def make_case(record: dict, *, case_id: str, suite: str, stratum: str, orientation: int,
              labels: dict[str, str] | None = None, groups: dict[str, str] | None = None) -> dict:
    """A case for a forward GenAIRR ``record``; ``orientation`` 1 presents the reverse complement of its read."""
    if orientation not in (0, 1):
        raise ValueError(f"{case_id}: orientation must be 0 or 1, got {orientation!r}")
    read = record["sequence"].upper()
    return {
        "case_id": case_id,
        "suite": suite,
        "stratum": stratum,
        "orientation": orientation,
        "sequence": revcomp(read) if orientation else read,
        "record": record,
        "labels": dict(labels or {}),
        "groups": dict(groups or {}),
    }


def write_cases(cases: Iterable[dict], path: str | Path) -> None:
    """Write one case per line; a ``.gz`` path is gzip-compressed."""
    write_jsonl(cases, path)


def read_cases(path: str | Path) -> list[dict]:
    """Every case of the file, in file order."""
    return read_jsonl(path)


def evenly_spaced(items: Sequence[T], n: int) -> list[T]:
    """``n`` of the N ``items``, evenly spaced: those at positions ``floor(i * N / n)`` for i from 0 to n - 1, in
    order (every (N / n)-th item when that is whole). A suite lists its cases stratum by stratum, so its first n
    cases would over-represent its first strata; an evenly spaced sample holds every stratum in about its share.
    ValueError unless n is from 1 to N."""
    if not 1 <= n <= len(items):
        raise ValueError(f"an evenly spaced sample takes 1 to {len(items)} of the {len(items)} items, got {n}")
    return [items[index * len(items) // n] for index in range(n)]


def write_fasta(cases: Iterable[dict], path: str | Path) -> None:
    """Write the presented reads as FASTA: ``>case_id`` then the sequence on one line."""
    with open(path, "w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(f">{case['case_id']}\n{case['sequence']}\n")


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """``(name, sequence)`` per FASTA record, in file order. The name is the header's first word; a sequence may
    span several lines, which are joined as written. Sequence text before the first header raises ValueError."""
    records: list[tuple[str, list[str]]] = []
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            line = line.strip()
            if line.startswith(">"):
                words = line[1:].split()
                records.append((words[0] if words else "", []))
            elif line:
                if not records:
                    raise ValueError(f"{path}:{number}: sequence before the first FASTA header")
                records[-1][1].append(line)
    return [(name, "".join(parts)) for name, parts in records]
