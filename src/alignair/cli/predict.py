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
                   help="fail the job if the AIRR-assembly failure rate exceeds this fraction "
                        "(default 0.01); assembly failures are always tagged per row regardless")
    p.add_argument("--permissive", action="store_true",
                   help="never fail on the AIRR-assembly failure rate (dirty repertoires); still tags rows")
    p.set_defaults(func=run)


def _load_metadata(args, ids):
    """Join a per-read metadata table (10x annotations / an AIRR TSV) to the read ids, returning
    ``(per_row_metas, extra_columns)`` for the writer, or ``(None, None)`` if no --metadata given."""
    if not args.metadata:
        return None, None
    from ..io.sequence_reader import load_metadata
    keep = [c.strip() for c in args.keep_columns.split(",")] if args.keep_columns else None
    meta, kept = load_metadata(args.metadata, id_column=args.metadata_id_column, keep_columns=keep)
    return [meta.get(i, {}) for i in ids], kept


def _write_rejects(path: str, rejects: list) -> None:
    """Write dropped/invalid input records so they are never silently lost."""
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sequence_id", "position", "reason", "sequence"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(rejects)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_run_metadata(out: str, model_path: str, args, n_reads: int, device: str,
                        failed: int = 0, fail_reasons: dict | None = None, loci=None,
                        partial: dict | None = None) -> None:
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
        "loci": list(loci) if loci else [],            # the model's exact locus mapping (P0-6)
        "command": {"input": args.input, "out": args.out, "columns": args.columns,
                    "locus": args.locus, "batch_size": args.batch_size, "device": device},
        "n_reads": n_reads, "offline": bool(args.offline),
        "airr_assembly_failed": failed, "airr_assembly_fail_reasons": fail_reasons or {},
        "airr_assembly_partial": partial or {},
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
    if args.rejects_out and stats.get("rejects"):        # emit dropped records instead of hiding them
        _write_rejects(args.rejects_out, stats["rejects"])
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
    fail_rate = len(failed) / len(records) if records else 0.0
    if failed and fail_rate > args.max_assembly_failures and not args.permissive:
        write_airr(args.out, ids, seqs, records, locus=out_locus, columns=args.columns)
        print(f"AIRR assembly failed for {len(failed)}/{len(records)} reads "
              f"({fail_rate:.1%} > --max-assembly-failures {args.max_assembly_failures:.1%}); "
              f"reasons: {dict(fail_reasons)}. Wrote tagged output to {args.out}. "
              f"Re-run with --permissive to accept, or raise --max-assembly-failures.")
        return 1

    metas, extra_cols = _load_metadata(args, ids)        # join per-read metadata (10x/AIRR) into output
    write_airr(args.out, ids, seqs, records, locus=out_locus, columns=args.columns,
               metas=metas, extra_columns=extra_cols)
    if not args.no_run_metadata:
        _write_run_metadata(args.out, model_path, args, len(records), device, failed=len(failed),
                            fail_reasons=dict(fail_reasons), loci=loci, partial=dict(partial))
    msg = f"aligned {len(records)} reads ({stats['n_dropped']} dropped) -> {args.out}"
    if failed or partial:
        msg += f"; {len(failed)} failed / {sum(partial.values())} partial AIRR assemblies tagged"
    print(msg)

    # a path, a pinned id/revision, or a direct HF repo: no catalog-update noise
    pinned = (os.path.exists(args.model) or "@" in args.model or bool(args.revision)
              or args.model.startswith("hf://"))
    maybe_notify_updates(sources_list=srcs, offline=args.offline, quiet=args.quiet, pinned=pinned)
    return 0
