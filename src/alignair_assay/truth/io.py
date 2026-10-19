"""JSON Lines (de)serialisation of observable truth records; ``.gz`` paths are gzip-compressed.

A truth file may start with a header line ``{"header": {...}}`` describing how it was built (the reference and
its digest); ``read_truth_jsonl`` skips it and ``read_truth_header`` returns it.
"""
from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from alignair_assay.truth.alleles import AlleleTruth
from alignair_assay.truth.boundaries import BoundaryTruth
from alignair_assay.truth.build import JunctionTruth, ObservableTruth, SegmentTruth
from alignair_assay.truth.segment import ObservedSegment


def truth_to_dict(t: ObservableTruth) -> dict:
    """JSON-ready dict: frozensets become sorted lists, tuples become lists."""
    return {
        "case_id": t.case_id,
        "stratum": t.stratum,
        "sequence": t.sequence,
        "orientation": t.orientation,
        "segments": {g: _segment_to_dict(s) for g, s in t.segments.items()},
        "unscorable": sorted(t.unscorable),
        "junction": asdict(t.junction) if t.junction is not None else None,
        "vj_in_frame": t.vj_in_frame,
        "stop_codon": t.stop_codon,
        "productive": t.productive,
        "clone_key": (None if t.clone_key is None
                      else [sorted(t.clone_key[0]), sorted(t.clone_key[1]), t.clone_key[2]]),
        "v_mismatches": t.v_mismatches,
        "v_frameshift": t.v_frameshift,
        "j_frameshift": t.j_frameshift,
        "excluded": t.excluded,
        "warnings": list(t.warnings),
    }


def truth_from_dict(d: dict) -> ObservableTruth:
    clone_key = d["clone_key"]
    return ObservableTruth(
        case_id=d["case_id"],
        stratum=d["stratum"],
        sequence=d["sequence"],
        orientation=d["orientation"],
        segments={g: _segment_from_dict(s) for g, s in d["segments"].items()},
        unscorable=frozenset(d["unscorable"]),
        junction=JunctionTruth(**d["junction"]) if d["junction"] is not None else None,
        vj_in_frame=d["vj_in_frame"],
        stop_codon=d["stop_codon"],
        productive=d["productive"],
        clone_key=(None if clone_key is None
                   else (frozenset(clone_key[0]), frozenset(clone_key[1]), clone_key[2])),
        v_mismatches=d["v_mismatches"],
        v_frameshift=d["v_frameshift"],
        j_frameshift=d["j_frameshift"],
        excluded=d["excluded"],
        warnings=tuple(d["warnings"]),
    )


def write_truth_jsonl(truths: Iterable[ObservableTruth], path: str | Path, header: dict | None = None) -> None:
    with _open(path, "wt") as fh:
        if header is not None:
            fh.write(json.dumps({"header": header}, separators=(",", ":")) + "\n")
        for t in truths:
            fh.write(json.dumps(truth_to_dict(t), separators=(",", ":")) + "\n")


def read_truth_jsonl(path: str | Path) -> list[ObservableTruth]:
    with _open(path, "rt") as fh:
        rows = (json.loads(line) for line in fh if line.strip())
        return [truth_from_dict(row) for row in rows if "header" not in row]


def read_truth_header(path: str | Path) -> dict | None:
    """The header of a truth file, or None when it has none."""
    with _open(path, "rt") as fh:
        for line in fh:
            if line.strip():
                return json.loads(line).get("header")
    return None


def _open(path: str | Path, mode: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def _segment_to_dict(s: SegmentTruth) -> dict:
    a = s.alleles
    return {
        "observed": asdict(s.observed),
        "boundaries": {k: list(v) for k, v in asdict(s.boundaries).items()},
        "alleles": {
            "primary": a.primary,
            "alleles": sorted(a.alleles),
            "genes": sorted(a.genes),
            "simulator_ties": sorted(a.simulator_ties),
            "disagrees_with_simulator": a.disagrees_with_simulator,
        },
        "observed_length": s.observed_length,
        "inverted": s.inverted,
    }


def _segment_from_dict(d: dict) -> SegmentTruth:
    a = d["alleles"]
    return SegmentTruth(
        observed=ObservedSegment(**d["observed"]),
        boundaries=BoundaryTruth(**{k: tuple(v) for k, v in d["boundaries"].items()}),
        alleles=AlleleTruth(
            primary=a["primary"],
            alleles=frozenset(a["alleles"]),
            genes=frozenset(a["genes"]),
            simulator_ties=frozenset(a["simulator_ties"]),
            disagrees_with_simulator=a["disagrees_with_simulator"],
        ),
        observed_length=d["observed_length"],
        inverted=d["inverted"],
    )
