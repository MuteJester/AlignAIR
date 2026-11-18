"""Golden records: a model's AIRR rows on a fixed read set, and the check that another code or model file gives the
same rows.

``record_golden`` predicts every read of a FASTA with a model file and writes the rows as JSON lines (a header with the
model's and the reads' SHA-256, then ``{"id", "sequence", "row"}`` per read); ``check_golden`` predicts the same reads
with a model file under the code in this process and names every field that differs. Strings, integers, booleans and
nulls must be equal; floats (and lists of floats) may differ by ``TOLERANCE``, the float noise a reorganised
computation leaves in likelihoods. Prediction runs in this process on one torch thread with batch 16, as a CPU
worker predicts, so a check is a pure function of the code and the model file.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field

from alignair_assay.core.cases import read_fasta

TOLERANCE = 1e-4
KIND = "alignair-golden"


@dataclass
class GoldenReport:
    """``n`` reads checked; ``differences``: (read id, field, expected, actual) for every field that differs."""

    n: int
    differences: list[tuple[str, str, object, object]] = field(default_factory=list)


def predict_rows(model_path: str, sequences: list[str]) -> list[dict]:
    """The AIRR rows ``Aligner.predict`` gives the sequences with the model file, on one torch thread, batch 16, as
    JSON gives them back (numbers, strings and lists; sorted keys)."""
    import torch

    from alignair.aligner import Aligner
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rows = Aligner.from_pretrained(model_path, device="cpu").predict(list(sequences), batch_size=16).to_dicts()
    finally:
        torch.set_num_threads(threads)
    return [json.loads(json.dumps(row, default=str, sort_keys=True)) for row in rows]


def _same(expected, actual, tolerance: float) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if isinstance(expected, int) and isinstance(actual, int):
            return expected == actual
        if math.isnan(expected) or math.isnan(actual):
            return math.isnan(expected) and math.isnan(actual)
        return abs(expected - actual) <= tolerance
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(_same(a, b, tolerance) for a, b in zip(expected, actual))
    return expected == actual


def compare_rows(expected: dict, actual: dict, tolerance: float = TOLERANCE) -> list[str]:
    """The fields of two rows that differ, sorted: a field one row lacks, or values that differ (floats beyond
    ``tolerance``, anything else at all)."""
    return sorted(key for key in expected.keys() | actual.keys()
                  if key not in expected or key not in actual or not _same(expected[key], actual[key], tolerance))


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record_golden(model_path: str, reads_fasta: str, out_path: str) -> int:
    """Write the model's rows of every read of ``reads_fasta`` to ``out_path`` (atomically); the number of reads."""
    reads = read_fasta(reads_fasta)
    rows = predict_rows(model_path, [sequence for _, sequence in reads])
    header = {"kind": KIND, "model_sha256": _sha256(model_path), "reads_sha256": _sha256(reads_fasta), "n": len(reads)}
    tmp = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp, "w") as out:
        out.write(json.dumps(header) + "\n")
        for (read_id, sequence), row in zip(reads, rows):
            out.write(json.dumps({"id": read_id, "sequence": sequence, "row": row}, sort_keys=True) + "\n")
    os.replace(tmp, out_path)
    return len(reads)


def read_golden(golden_path: str) -> tuple[dict, list[dict]]:
    """``(header, records)`` of a golden file; ValueError for a file that is not one."""
    with open(golden_path) as handle:
        header, *records = (json.loads(line) for line in handle if line.strip())
    if header.get("kind") != KIND or header.get("n") != len(records):
        raise ValueError(f"{golden_path} is not a golden record file (kind {KIND!r} and its n reads)")
    return header, records


def check_golden(model_path: str, golden_path: str, tolerance: float = TOLERANCE) -> GoldenReport:
    """Predict the golden file's reads with ``model_path`` and name every field that differs from the recorded rows."""
    _, records = read_golden(golden_path)
    rows = predict_rows(model_path, [record["sequence"] for record in records])
    report = GoldenReport(n=len(records))
    for record, row in zip(records, rows):
        for key in compare_rows(record["row"], row, tolerance):
            report.differences.append((record["id"], key, record["row"].get(key), row.get(key)))
    return report
