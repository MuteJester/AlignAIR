"""Command-line adapters for reproducible guard training studies."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from alignair.errors import (
    AlignAIRError, DataContractError, InternalError, MissingDependencyError,
    OperationCancelledError, OperationIOError, report_error,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--error-format", choices=("text", "json"), default="text")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-sweep", help="bind explicit sweep settings to development data")
    for option in ("settings", "records", "profile-manifest", "fresh-sources", "checkpoints", "out"):
        freeze.add_argument("--" + option, type=Path, required=True)
    freeze_loco = commands.add_parser("freeze-loco", help="bind LOCO folds to a frozen sweep and its development results")
    for option in ("training-manifest", "runs", "records", "profile-manifest", "checkpoints", "out"):
        freeze_loco.add_argument("--" + option, type=Path, required=True)
    for command in ("sweep", "loco"):
        child = commands.add_parser(command)
        child.add_argument("--manifest", type=Path, required=True, help=f"frozen {command} manifest JSON")
        if command == "loco":
            child.add_argument("--training-manifest", type=Path, required=True)
        child.add_argument("--records", type=Path, required=True, help="development records JSONL")
        child.add_argument("--profiles", type=Path, required=True, help="directory of profile HMM files")
        child.add_argument("--out", type=Path, required=True, help="study report and accounting directory")
        child.add_argument("--checkpoints", type=Path, help="override the manifest's checkpoint output directory")
        child.add_argument("--feature-cache", type=Path, required=True,
                           help="cache JSON file for sweep; cache directory for LOCO")
        child.add_argument("--device", default="auto", help="auto, cpu, cuda, or a Torch device")
    return parser


def _read_json(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise DataContractError("Training input must be a JSON mapping.", context={"path": path})
    return value


def _read_records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _execute(args):
    if args.command == "freeze-loco":
        from .manifests import freeze_loco_manifest

        manifest = freeze_loco_manifest(
            _read_records(args.records), output_path=args.out, training_manifest=_read_json(args.training_manifest),
            run_results=_read_json(args.runs), bank_checksum=_read_json(args.profile_manifest).get("bank_checksum"),
            checkpoint_dir=args.checkpoints,
            input_files=(args.records, args.training_manifest, args.runs, args.profile_manifest),
        )
        print(json.dumps({"expected_models": manifest["expected_models"], "frozen_seeds": manifest["frozen_seeds"],
                          "manifest_sha256": manifest["manifest_sha256"]}, indent=2, sort_keys=True))
        return 0
    if args.command == "freeze-sweep":
        from .manifests import freeze_sweep_manifest

        manifest = freeze_sweep_manifest(
            _read_records(args.records), output_path=args.out, settings=_read_json(args.settings),
            bank_checksum=_read_json(args.profile_manifest).get("bank_checksum"),
            checkpoint_dir=args.checkpoints,
            forbidden_sequence_hashes=_fresh_sequence_hashes(args.fresh_sources),
        )
        print(json.dumps({key: manifest[key] for key in (
            "expected_runs", "n_train", "n_dev", "train_partition_hash",
            "forbidden_fresh_seqhashes_count", "forbidden_leak_into_dev_by_sequence", "manifest_sha256",
        )}, indent=2, sort_keys=True))
        return 0
    from alignair.aa.guard.profiles import load_bank
    from .engine import resolve_device
    from .studies import run_loco, run_sweep

    device = resolve_device(args.device)
    manifest = _read_json(args.manifest)
    training_manifest = _read_json(args.training_manifest) if args.command == "loco" else None
    records = _read_records(args.records)
    bank = load_bank(args.profiles)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "sweep":
        results = run_sweep(manifest, records, bank, artifact_dir=args.out,
                            cache_path=args.feature_cache, device=device, checkpoint_dir=args.checkpoints)["runs"]
    else:
        results = run_loco(training_manifest, manifest, records, bank, artifact_dir=args.out,
                           cache_dir=args.feature_cache, device=device, checkpoint_dir=args.checkpoints)
    failed = sum(result["status"] != "ok" for result in results.values())
    print(json.dumps({"runs": len(results), "failed": failed, "output": str(args.out)}))
    return 1 if failed else 0


def _fresh_sequence_hashes(directory):
    """Read every supplied source; malformed or absent exclusions must stop freezing."""
    paths = sorted(directory.glob("*.normalized.json"))
    if not paths:
        raise DataContractError("Fresh exclusion source directory contains no normalized JSON files.",
                                context={"path": directory})
    hashes = set()
    for path in paths:
        records = json.loads(path.read_text())
        if not isinstance(records, list):
            raise DataContractError("Fresh exclusion source must contain a record list.", context={"path": path})
        for index, record in enumerate(records):
            if not isinstance(record, dict) or not isinstance(record.get("aa"), str) or not record["aa"]:
                raise DataContractError("Fresh exclusion record requires a text sequence.",
                                        context={"path": path, "record_index": index})
            hashes.add(hashlib.sha256(record["aa"].encode()).hexdigest())
    return hashes


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return _execute(args)
    except AlignAIRError as exc:
        error = exc
    except KeyboardInterrupt:
        error = OperationCancelledError("Guard training interrupted.")
    except OSError as exc:
        error = OperationIOError("Cannot access guard training inputs or outputs.", context={"reason": str(exc)})
        error.__cause__ = exc
    except json.JSONDecodeError as exc:
        error = DataContractError("Invalid JSON in guard training input.", context={"line": exc.lineno})
        error.__cause__ = exc
    except ModuleNotFoundError as exc:
        error = MissingDependencyError("Guard training requires an optional research dependency.",
                                       context={"module": exc.name})
        error.__cause__ = exc
    except Exception as exc:
        error = InternalError("Unexpected guard training failure.", context={"cause_type": type(exc).__name__})
        error.__cause__ = exc
    return report_error(error, debug=args.debug, json_output=args.error_format == "json")
