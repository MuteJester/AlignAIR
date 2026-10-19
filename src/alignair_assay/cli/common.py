"""Pieces shared by the commands: argument types, run provenance, matching summaries and report files."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from alignair_assay.core.provenance import git_state, sha256_file
from alignair_assay.metrics.case import MIN_OBSERVED
from alignair_assay.report import to_markdown

DEFAULT_REFERENCE = "HUMAN_IGH_OGRDB"


def file_name(value: str) -> str:
    """Argument type for a name used as a file-name component (a tool name): no directory part."""
    if not value or Path(value).name != value:
        raise argparse.ArgumentTypeError(f"not a plain file-name component: {value!r}")
    return value


def provenance(command: str, inputs: dict[str, str], **extra) -> dict:
    """Provenance of a command run: the command, each input's path and SHA-256, the assay checkout's git state, a
    UTC timestamp, the minimum observed lengths in force, and ``extra``."""
    return {
        "command": command,
        "inputs": {k: {"path": str(p), "sha256": sha256_file(p)} for k, p in inputs.items()},
        **git_state(Path(__file__).resolve().parent),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "min_observed": dict(MIN_OBSERVED),
        **extra,
    }


def print_matching(tool: str, m: dict) -> None:
    """One line of prediction/truth matching counts for ``tool`` (see ``scoring.pipeline.score_tool``)."""
    print(f"{tool}: {m['scored']} cases scored ({m['excluded']} excluded of {m['truth_cases']}); "
          f"{m['rows']} rows, {m['duplicates']} duplicates, {m['invalid_rows']} invalid rows; "
          f"missing {m['missing']}, extra {m['extra']}, ignored for excluded cases {m['excluded_ignored']}")


def write_report(report: dict, json_path: Path, md_path: Path) -> None:
    """Write a report as strict JSON and as its markdown summary."""
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
