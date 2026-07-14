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

from ..aligner import Aligner
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
    p.add_argument("--sequence-column", default=None, help="sequence column name (CSV/TSV input)")
    p.add_argument("--id-column", default=None, help="id column name (CSV/TSV input)")
    p.add_argument("--metadata", default=None,
                   help="per-read metadata table (e.g. 10x filtered_contig_annotations.csv / an AIRR TSV) "
                        "to join by id and carry into output (cell_id/barcode/umi_count/…)")
    p.add_argument("--metadata-id-column", default=None, help="id column in the --metadata table")
    p.add_argument("--keep-columns", default=None,
                   help="comma-separated metadata columns to carry through (default: known 10x/AIRR set)")
    p.add_argument("--rejects-out", default=None,
                   help="write dropped/invalid input records here (id, position, reason, sequence)")
    p.add_argument("--locus", default=None,
                   help="locus label for records the model can't attribute (default: the model's locus "
                        "if it declares exactly one; else IGH). Multi-chain records carry their own locus.")
    p.add_argument("--columns", default=None,
                   help="output columns: a preset (full/core/minimal/airr) or a comma-separated "
                        "field list (default: full). Light selections skip the AIRR assembly for speed.")
    p.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--genotype", default=None,
                   help="genotype JSON/YAML {gene: [alleles]} (a subset of the trained reference) to "
                        "constrain the allele calls to")
    p.add_argument("--genotype-method", choices=["mask", "softmax", "renormalize", "redistribute"],
                   default="mask", help="how to apply the genotype constraint (default: mask)")
    p.add_argument("--registry", action="append", help="registry url for a shipped model id (repeatable)")
    p.add_argument("--revision", default=None,
                   help="pin a Hugging Face branch/tag/commit (for hf://org/repo) or a catalog version")
    p.add_argument("--hf-token", default=None,
                   help="Hugging Face token for a private/gated repo (falls back to $HF_TOKEN)")
    p.add_argument("--offline", action="store_true", help="never touch the network")
    p.add_argument("--quiet", action="store_true", help="suppress the update-available notice")
    p.add_argument("--trust-pickle", action="store_true",
                   help="allow loading a legacy .pt / pre-safe .alignair (runs pickle); path-only")
    p.add_argument("--no-run-metadata", action="store_true", help="do not write the <out>.run.json sidecar")
    p.add_argument("--max-assembly-failures", type=float, default=0.01,
                   help="fail the job if the AIRR-assembly FAILURE rate exceeds this fraction "
                        "(default 0.01); failures are always tagged per row regardless")
    p.add_argument("--max-partial-assemblies", type=float, default=1.0,
                   help="fail the job if the PARTIAL-assembly rate exceeds this fraction "
                        "(default 1.0 = never; partial rows are always tagged with a reason)")
    p.add_argument("--permissive", action="store_true",
                   help="never fail on assembly failure/partial rates (dirty repertoires); still tags rows")
    p.set_defaults(func=run)


def _load_metadata(args, ids):
    """Join a per-read metadata table (10x annotations / an AIRR TSV) to the read ids, returning
    ``(per_row_metas, extra_columns)`` for the writer, or ``(None, None)`` if no --metadata given.

    Metadata may NEVER overwrite a model/scientific AIRR field: any metadata column whose name collides
    with a produced field (sequence/v_call/productive/coords/…) is namespaced to ``meta_<col>`` so the
    aligner's result is preserved (AIRR-review). 10x column names are normalized to AIRR."""
    if not args.metadata:
        return None, None
    from ..io.airr import COLUMNS
    from ..io.sequence_reader import load_metadata
    keep = [c.strip() for c in args.keep_columns.split(",")] if args.keep_columns else None
    meta, kept = load_metadata(args.metadata, id_column=args.metadata_id_column, keep_columns=keep,
                               normalize_10x=True)
    protected = frozenset(COLUMNS)

    def _safe(col):                                # protect model fields from metadata clobbering
        return f"meta_{col}" if col in protected else col
    kept_safe = [_safe(c) for c in kept]
    metas = [{_safe(k): v for k, v in meta.get(i, {}).items()} for i in ids]
    return metas, kept_safe


def _write_rejects(path: str, rejects: list) -> None:
    """Write dropped/invalid input records (0-based `position`) so they are never silently lost. Written
    atomically; a valid header-only table is emitted even when nothing was rejected."""
    import csv
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sequence_id", "position", "reason", "sequence"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(rejects or [])
    os.replace(tmp, path)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_run_metadata(out: str, model_path: str, args, device: str, stats: dict, records: list, *,
                        loci=None, failed: int = 0, fail_reasons: dict | None = None,
                        partial: dict | None = None) -> None:
    from ..model_file import container, read_metadata
    md = read_metadata(model_path) if container.is_alignair_file(model_path) else {}
    ref = md.get("reference", {})
    n = len(records)
    complete = sum(1 for r in records if r.get("airr_assembly_status") == "complete")
    partial_n = sum(1 for r in records if r.get("airr_assembly_status") == "partial")
    cropped = sum(1 for r in records if r.get("length_cropped"))
    prov = {
        "model_spec": args.model, "model_path": os.path.abspath(model_path),
        "model_id": md.get("model_id"), "model_version": md.get("model_version"),
        "artifact_sha256": _sha256_file(model_path),
        "reference_fasta_sha256": ref.get("reference_fasta_sha256"),
        "allele_order_sha256": ref.get("allele_order_sha256"),
        "alignair_version": md.get("created_by_alignair"),
        "loci": list(loci) if loci else [],            # the model's exact locus mapping (P0-6)
        "command": {"input": args.input, "out": args.out, "columns": args.columns,
                    "locus": args.locus, "batch_size": args.batch_size, "device": device},
        "counts": {                                    # full record accounting (AIRR-review item 5)
            "input_records": stats.get("n_read", n + stats.get("n_dropped", 0)),
            "accepted_records": n, "rejected_records": stats.get("n_dropped", 0),
            "cropped_records": cropped, "complete_assemblies": complete,
            "partial_assemblies": partial_n, "failed_assemblies": failed, "written_records": n,
        },
        "airr_assembly_fail_reasons": fail_reasons or {}, "airr_assembly_partial_reasons": partial or {},
        "offline": bool(args.offline),
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
    ids, seqs, stats = read_sequences(args.input, seq_column=args.sequence_column,
                                      id_column=args.id_column, collect_rejects=bool(args.rejects_out))
    if args.rejects_out:                                 # emit dropped records (header-only if none)
        _write_rejects(args.rejects_out, stats.get("rejects") or [])
    if not seqs:
        print(f"no valid reads in {args.input}")
        return 1

    srcs = _sources.resolve_sources(args.registry)
    try:
        model_path = str(resolve_model(args.model, sources=srcs, offline=args.offline,
                                       token=args.hf_token, revision=args.revision))
    except Exception as e:
        print(f"could not resolve model '{args.model}': {e}")
        return 1
    is_alignair = container.is_alignair_file(model_path)
    if not is_alignair and not args.dataconfig:
        print("--dataconfig is required for legacy .pt models")
        return 1
    try:                        # the CLI is a thin client of the stable Aligner API (P0-9)
        aligner = Aligner.from_pretrained(model_path, device=device, dataconfigs=args.dataconfig,
                                          trust_pickle=args.trust_pickle)
    except ValueError as e:
        print(str(e))
        return 1
    reference = aligner.reference

    # locus: don't silently default to IGH when the model declares its locus (P0-6). Validate an
    # explicit --locus against the model's loci; otherwise use the model's single locus if it has one.
    loci = reference.locus_names() if hasattr(reference, "locus_names") else ()
    if args.locus and loci and args.locus.upper() not in {l.upper() for l in loci}:
        print(f"--locus {args.locus} is not one of this model's loci {loci}")
        return 1
    out_locus = args.locus or (loci[0] if len(loci) == 1 else "IGH")

    overrides = {}
    if args.genotype:
        from ..genotype.constraint import load_genotype
        try:                    # fixed-head models cannot call novel alleles -> reject, do not drop
            genotype = load_genotype(args.genotype, reference=reference, drop_unknown=False)
        except ValueError as e:
            print(f"invalid genotype: {e}")
            return 1
        overrides = {"genotype": genotype, "genotype_method": args.genotype_method}
    records = aligner.predict(seqs, batch_size=args.batch_size,
                              airr=needs_assembly(args.columns), **overrides).to_dicts()

    # AIRR-assembly accounting by explicit state (complete / partial / failed): rows are always tagged;
    # fail the job only if the *failed* rate is too high (unless --permissive), so a run cannot silently
    # lose junction/region/alignment fields. Partial records are reported but not a hard failure (P0-7 /
    # AIRR-review #5).
    from collections import Counter
    failed = [r for r in records if r.get("airr_assembly_status") == "failed"]
    partial = Counter(r.get("airr_assembly_reason", "?") for r in records
                      if r.get("airr_assembly_status") == "partial")
    fail_reasons = Counter(r.get("airr_assembly_error", "?").split(":")[0] for r in failed)
    n = len(records)
    fail_rate = len(failed) / n if n else 0.0
    partial_rate = sum(partial.values()) / n if n else 0.0
    metas, extra_cols = _load_metadata(args, ids)        # join per-read metadata (10x/AIRR) into output

    def _finish():                                       # write the tagged output (always) + provenance
        write_airr(args.out, ids, seqs, records, locus=out_locus, columns=args.columns,
                   metas=metas, extra_columns=extra_cols)
        if not args.no_run_metadata:
            _write_run_metadata(args.out, model_path, args, device, stats, records, loci=loci,
                                failed=len(failed), fail_reasons=dict(fail_reasons), partial=dict(partial))

    if not args.permissive and failed and fail_rate > args.max_assembly_failures:
        _finish()
        print(f"AIRR assembly FAILED for {len(failed)}/{n} reads ({fail_rate:.1%} > "
              f"--max-assembly-failures {args.max_assembly_failures:.1%}); reasons {dict(fail_reasons)}. "
              f"Wrote tagged output to {args.out}. Re-run with --permissive or raise the threshold.")
        return 1
    if not args.permissive and partial and partial_rate > args.max_partial_assemblies:
        _finish()
        print(f"AIRR assembly PARTIAL for {sum(partial.values())}/{n} reads ({partial_rate:.1%} > "
              f"--max-partial-assemblies {args.max_partial_assemblies:.1%}); reasons {dict(partial)}. "
              f"Wrote tagged output to {args.out}. Re-run with --permissive or raise the threshold.")
        return 1

    _finish()
    msg = f"aligned {n} reads ({stats['n_dropped']} dropped) -> {args.out}"
    if failed or partial:
        msg += f"; {len(failed)} failed / {sum(partial.values())} partial AIRR assemblies tagged"
    print(msg)

    # a path, a pinned id/revision, or a direct HF repo: no catalog-update noise
    pinned = (os.path.exists(args.model) or "@" in args.model or bool(args.revision)
              or args.model.startswith("hf://"))
    maybe_notify_updates(sources_list=srcs, offline=args.offline, quiet=args.quiet, pinned=pinned)
    return 0
