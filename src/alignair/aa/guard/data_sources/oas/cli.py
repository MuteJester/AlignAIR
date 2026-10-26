"""Command-line adapters for explicit OAS collection inputs and outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignair.aa.guard.disjoint import build_gate3_global_denylist
from alignair.errors import (
    AlignAIRError, DataContractError, InternalError, OperationCancelledError, OperationIOError, report_error,
)
from .collectors import DiversityPolicy, collect_paired, collect_unpaired
from .probe import probe_species
from .records import Exclusions


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--error-format", choices=("text", "json"), default="text")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("probe", "unpaired", "paired", "diverse"):
        child = commands.add_parser(command)
        child.add_argument("--cleaned", type=Path, required=True, help="pre-split cohort JSONL for exclusions")
        if command in ("probe", "unpaired"):
            child.add_argument("--catalog", type=Path, required=True, help="species/completeness catalog JSON")
        if command != "probe":
            child.add_argument("--fingerprint", type=Path, required=True, help="frozen pre-split cohort fingerprint")
            child.add_argument("--phase3-gold", type=Path, required=True, help="Phase-3 labels TSV for digest replay")
            child.add_argument("--store", type=Path, required=True, help="raw cache and normalized sequence directory")
            child.add_argument("--census-dir", type=Path, required=True, help="counts-only report directory")
    return parser


def _execute(args):
    records = []
    with args.cleaned.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise DataContractError("Pre-split cohort records must be JSON mappings.", context={"line": line_number})
            if args.command != "probe":
                for field in ("aa", "group", "category", "lineage_key"):
                    if not isinstance(record.get(field), str):
                        raise DataContractError("Pre-split cohort record is missing a text field.",
                                                context={"line": line_number, "field": field})
            records.append(record)
    if args.command == "probe":
        return probe_species(Exclusions.from_records(records, {}).studies, catalog_path=args.catalog)
    # Require the explicit replay input rather than falling back to a local notes path.
    if not args.phase3_gold.is_file():
        raise OperationIOError("Phase-3 digest replay labels are missing.", context={"path": args.phase3_gold})
    fingerprint = json.loads(args.fingerprint.read_text())
    if not isinstance(fingerprint, dict) or not isinstance(fingerprint.get("record_digest", {}), dict):
        raise DataContractError("Pre-split fingerprint requires a record_digest mapping.")
    denylist = build_gate3_global_denylist(args.cleaned, fingerprint, phase3_gold_path=args.phase3_gold)
    exclusions = Exclusions.from_records(records, denylist)
    options = {"store": args.store, "census_dir": args.census_dir}
    if args.command == "unpaired":
        result = collect_unpaired(exclusions, catalog_bytes=args.catalog.read_bytes(), **options)
    elif args.command == "diverse":
        result = collect_paired(exclusions, policy=DiversityPolicy(), **options)
    else:
        result = collect_paired(exclusions, **options)
    return result.census


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        print(json.dumps(_execute(args), indent=2, sort_keys=True))
        return 0
    except AlignAIRError as exc:
        error = exc
    except KeyboardInterrupt:
        error = OperationCancelledError("OAS data collection interrupted.")
    except OSError as exc:
        error = OperationIOError("Cannot access OAS collection inputs or outputs.", context={"reason": str(exc)})
        error.__cause__ = exc
    except json.JSONDecodeError as exc:
        error = DataContractError("Invalid JSON in OAS collection input.", context={"line": exc.lineno})
        error.__cause__ = exc
    except Exception as exc:
        error = InternalError("Unexpected OAS collection failure.", context={"cause_type": type(exc).__name__})
        error.__cause__ = exc
    return report_error(error, debug=args.debug, json_output=args.error_format == "json")
