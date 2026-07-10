"""``alignair predict`` — align reads with a trained model and write an AIRR TSV.

``--model`` accepts a filesystem path OR a shipped registry id / ``id@version`` (resolved + verified
into the local cache). Writes an ``<out>.run.json`` provenance sidecar and passively notes if a newer
model is available (suppressed for a pinned id / a path / ``--offline`` / ``--quiet``).
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

from ..api import load_model, predict_sequences
from ..io.airr import write_airr
from ..io.sequence_reader import read_sequences


def register(sub) -> None:
    p = sub.add_parser("predict", help="align reads with a trained model -> AIRR TSV")
    p.add_argument("--model", required=True,
                   help="trained model: a .alignair/.pt path, or a shipped registry id / id@version")
    p.add_argument("--dataconfig", nargs="+", default=None,
                   help="GenAIRR dataconfig(s) for the germline reference; optional for a .alignair "
                        "model (it carries its own), required for a legacy .pt")
    p.add_argument("--input", required=True, help="reads to align (FASTA / FASTQ / TSV; .gz ok; '-' = stdin)")
    p.add_argument("--out", required=True, help="output AIRR TSV path")
    p.add_argument("--locus", default="IGH", help="locus label for the AIRR output")
    p.add_argument("--columns", default=None,
                   help="output columns: a preset (full/core/minimal/airr) or a comma-separated "
                        "field list (default: full). Light selections skip the AIRR assembly for speed.")
    p.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--registry", action="append", help="registry url for a shipped model id (repeatable)")
    p.add_argument("--offline", action="store_true", help="never touch the network")
    p.add_argument("--quiet", action="store_true", help="suppress the update-available notice")
    p.add_argument("--trust-pickle", action="store_true",
                   help="allow loading a legacy .pt / pre-safe .alignair (runs pickle); path-only")
    p.add_argument("--no-run-metadata", action="store_true", help="do not write the <out>.run.json sidecar")
    p.set_defaults(func=run)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_run_metadata(out: str, model_path: str, args, n_reads: int, device: str) -> None:
    from ..model_file import container, read_metadata
    md = read_metadata(model_path) if container.is_alignair_file(model_path) else {}
    ref = md.get("reference", {})
    prov = {
        "model_spec": args.model, "model_path": os.path.abspath(model_path),
        "model_id": md.get("model_id"), "model_version": md.get("model_version"),
        "artifact_sha256": _sha256_file(model_path),
        "reference_fasta_sha256": ref.get("reference_fasta_sha256"),
        "allele_order_sha256": ref.get("allele_order_sha256"),
        "alignair_version": md.get("created_by_alignair"),
        "command": {"input": args.input, "out": args.out, "columns": args.columns,
                    "locus": args.locus, "batch_size": args.batch_size, "device": device},
        "n_reads": n_reads, "offline": bool(args.offline),
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    with open(out + ".run.json", "w") as f:
        json.dump(prov, f, indent=2)


def run(args) -> int:
    import torch
    from ..io.airr import needs_assembly
    from ..model_file import container
    from ..registry import maybe_notify_updates, resolve_model, sources as _sources
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ids, seqs, stats = read_sequences(args.input)
    if not seqs:
        print(f"no valid reads in {args.input}")
        return 1

    srcs = _sources.resolve_sources(args.registry)
    try:
        model_path = str(resolve_model(args.model, sources=srcs, offline=args.offline))
    except Exception as e:
        print(f"could not resolve model '{args.model}': {e}")
        return 1
    is_alignair = container.is_alignair_file(model_path)
    if not is_alignair and not args.dataconfig:
        print("--dataconfig is required for legacy .pt models")
        return 1
    try:
        model, reference = load_model(model_path, dataconfigs=args.dataconfig, device=device,
                                      trust_pickle=args.trust_pickle)
    except ValueError as e:
        print(str(e))
        return 1

    records = predict_sequences(model, reference, seqs, device=device, batch_size=args.batch_size,
                                airr=needs_assembly(args.columns))
    write_airr(args.out, ids, seqs, records, locus=args.locus, columns=args.columns)
    if not args.no_run_metadata:
        _write_run_metadata(args.out, model_path, args, len(records), device)
    print(f"aligned {len(records)} reads ({stats['n_dropped']} dropped) -> {args.out}")

    pinned = os.path.exists(args.model) or "@" in args.model        # a path or a pinned id: no update noise
    maybe_notify_updates(sources_list=srcs, offline=args.offline, quiet=args.quiet, pinned=pinned)
    return 0
