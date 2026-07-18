"""JSONL persistence for benchmark cases."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..core.schema import BenchmarkCase


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def save_jsonl(cases: Iterable[BenchmarkCase], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case.to_dict(), sort_keys=True, default=_json_default) + "\n")


def load_jsonl(path: str | Path) -> list[BenchmarkCase]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [BenchmarkCase.from_dict(json.loads(line)) for line in f if line.strip()]


def save_dicts_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    """Save dictionaries as JSONL."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def load_dicts_jsonl(path: str | Path) -> list[dict]:
    """Load dictionaries from JSONL."""

    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
