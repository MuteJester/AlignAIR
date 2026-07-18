"""Shared helpers for CLI command handlers."""
from __future__ import annotations

import json


def emit_json(payload: dict, out_path: str | None = None) -> None:
    """Print a JSON payload or write it to a file."""

    text = json.dumps(payload, indent=2, sort_keys=True)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    else:
        print(text)
