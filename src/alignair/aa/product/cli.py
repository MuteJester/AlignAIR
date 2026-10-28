"""Protein inference CLI: raw sequence files -> predictions and a provenance sidecar.

Shared ``alignair.io`` parsing handles FASTA/FASTQ, CSV/TSV, text, gzip and stdin. Protein
sequence normalization is applied downstream by ``AAPredictor.predict``; nucleotide cleaning
is never applied. The in-memory prediction facade continues to accept sequences, not paths.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from .loader import load_aa_model
from .predictor import _provenance
from ...data.inputs import SequenceInputFailure
from ...io import iter_sequence_records


def register(sub) -> None:
    p = sub.add_parser("predict-aa", help="amino-acid inference product: sequence files -> JSONL predictions")
    p.add_argument("input", help="reads to predict (FASTA, FASTQ, CSV/TSV or text; "
                                 "'-' = stdin; '.gz' ok)")
    p.add_argument("--model", default="alignair-aa-igh-human@2.0.0",
                   help="AA catalog id, Hub repo or .alignair path (default: alignair-aa-igh-human@2.0.0)")
    p.add_argument("--registry", action="append", help="registry source (repeatable)")
    p.add_argument("--revision", help="catalog version or Hub branch/tag/commit")
    p.add_argument("--offline", action="store_true", help="use local artifacts and caches only")
    p.add_argument("--token", help="Hugging Face access token (or set HF_TOKEN)")
    p.add_argument("--coverage-group", choices=["full", "trunc", "short"], default=None,
                   help="§7 Mondrian calibration group to apply to every record "
                        "(default: pooled fallback -- see §7)")
    p.add_argument("--on-invalid", choices=["error", "skip"], default="skip",
                   help="error = fail fast on invalid sequence; skip = emit per-record error (default: skip)")
    p.add_argument("--device", default="auto", help="cpu / cuda / auto (default: auto)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="maximum valid records per model forward (default: %(default)s)")
    p.add_argument("--out", required=True, help="output path: one §6 AAPrediction JSON object per line")
    p.add_argument("--no-run-metadata", action="store_true",
                   help="do not write the <out>.run.json provenance sidecar")
    p.set_defaults(func=run)


def _read_records(path: str) -> list[dict]:
    """Adapt shared raw records; the predictor assigns missing IDs and validates content."""
    return [record._asdict() for record in iter_sequence_records(path, strip_sequence_lines=True)]


def _write_run_metadata(out: str, predictor, args, counts: dict) -> None:
    prov = _provenance(predictor.card)
    meta = {
        **prov,
        "n_input": counts["n_input"],
        "n_predicted": counts["n_predicted"],
        "n_errors": counts["n_errors"],
        "command": {
            "input": args.input, "out": args.out, "model": args.model,
            "coverage_group": args.coverage_group, "on_invalid": args.on_invalid, "device": args.device,
            "revision": args.revision, "offline": args.offline,
            "batch_size": args.batch_size,
        },
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    with open(out + ".run.json", "w") as f:
        json.dump(meta, f, indent=2)


def run(args) -> int:
    records = _read_records(args.input)
    if not records:
        print(f"no sequences found in {args.input}")
        return 1

    predictor = load_aa_model(args.model, device=args.device, registry=args.registry,
                              revision=args.revision, offline=args.offline, token=args.token)
    # Pass through the on_invalid flag from the CLI (default: "skip" for batch-friendly mode).
    results = predictor.predict(records, coverage_group=args.coverage_group, on_invalid=args.on_invalid,
                                 batch_size=args.batch_size)

    n_errors = 0
    with open(args.out, "w") as f:
        for r in results:
            if isinstance(r, SequenceInputFailure):
                n_errors += 1
            f.write(json.dumps(r.to_dict()) + "\n")

    counts = {"n_input": len(records), "n_predicted": len(results) - n_errors, "n_errors": n_errors}
    if not args.no_run_metadata:
        _write_run_metadata(args.out, predictor, args, counts)

    print(f"predicted {counts['n_predicted']} sequences ({counts['n_errors']} errors) -> {args.out}")
    return 0
