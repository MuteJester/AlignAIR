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
    p.add_argument("--chunk-size", type=int, default=20000,
                   help="stream the input in chunks of this many reads (bounded memory; default 20000)")
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


def _load_metadata_table(args):
    """Load the per-read metadata JOIN TABLE once (id -> {col: value}), returning
    ``(meta_by_id, extra_columns)`` or ``(None, None)`` if no --metadata given. The per-chunk join looks
    up by id, so memory is bounded by the metadata table, not the repertoire.

    Metadata may NEVER overwrite a model/scientific AIRR field: any metadata column colliding with a
    produced field (sequence/v_call/productive/coords/…) is namespaced to ``meta_<col>`` so the aligner's
    result is preserved (AIRR-review). 10x column names are normalized to AIRR."""
    if not args.metadata:
        return None, None
    from ..io.airr import COLUMNS, _METADATA_FILL_ONLY
    from ..io.sequence_reader import build_metadata_index
    keep = [c.strip() for c in args.keep_columns.split(",")] if args.keep_columns else None
    protected = frozenset(COLUMNS)

    def _safe(col):                                # protect model fields from metadata clobbering
        if col in _METADATA_FILL_ONLY:             # keep AIRR name (c_call); writer fills only when blank
            return col
        return f"meta_{col}" if col in protected else col
    index, kept_safe = build_metadata_index(args.metadata, id_column=args.metadata_id_column,
                                            keep_columns=keep, normalize_10x=True, rename=_safe)
    return index, kept_safe


def _lookup_metas(meta_by_id, cids):
    """Per-chunk metadata rows aligned to ``cids``. Uses a single batched SQLite query when the index
    supports it (bounded, not one query per read); falls back to a plain dict for small/eager callers."""
    if meta_by_id is None:
        return None
    found = meta_by_id.get_many(cids) if hasattr(meta_by_id, "get_many") else meta_by_id
    return [found.get(i, {}) for i in cids]


def _stream_predict(aligner, *, input_path, out_path, columns, chunk_size, seq_column, id_column,
                    meta_by_id, extra_cols, out_locus, overrides, batch_size, rejects_out):
    """Stream reader-chunk -> predict -> AIRR assembly -> metadata join -> writer -> counters, in
    bounded memory (peak ~ chunk_size, not repertoire size). Order and cross-chunk duplicate-id handling
    are preserved by the reader; rejects are written incrementally (never accumulated). Returns
    ``(counts, fail_reasons, partial_reasons)``. The output is committed atomically on a clean pass."""
    import csv
    from collections import Counter

    from ..io.airr import AirrWriter, needs_assembly
    from ..io.sequence_reader import iter_sequences
    do_airr = needs_assembly(columns)
    counts = dict(input=0, accepted=0, rejected=0, cropped=0, complete=0, partial=0, failed=0,
                  written=0, nonstandard_orientation=0)
    fail_reasons, partial_reasons = Counter(), Counter()

    rejects_buf = [] if rejects_out else None
    rej_f = rej_w = rej_tmp = None
    if rejects_out:                                  # stream rejects to a temp file, rename atomically
        rej_tmp = f"{rejects_out}.tmp.{os.getpid()}"
        rej_f = open(rej_tmp, "w", newline="")
        rej_w = csv.DictWriter(rej_f, fieldnames=["sequence_id", "position", "reason", "sequence"],
                               delimiter="\t")
        rej_w.writeheader()

    writer = AirrWriter(out_path, locus=out_locus, columns=columns, extra_columns=extra_cols)
    committed = False
    try:
        for cids, cseqs, dropped in iter_sequences(input_path, chunk_size=chunk_size,
                                                   seq_column=seq_column, id_column=id_column,
                                                   rejects=rejects_buf):
            counts["rejected"] += dropped
            if rejects_buf is not None:              # drain this chunk's rejects to disk, keep memory flat
                for rj in rejects_buf:
                    rej_w.writerow(rj)
                rejects_buf.clear()
            if not cseqs:
                continue
            records = aligner.predict(cseqs, batch_size=batch_size, airr=do_airr, **overrides).to_dicts()
            metas = _lookup_metas(meta_by_id, cids)  # ONE indexed query per chunk, not per read
            writer.write(cids, cseqs, records, metas=metas)
            counts["accepted"] += len(cseqs)
            counts["written"] += len(records)
            for r in records:
                st = r.get("airr_assembly_status")
                if st == "complete":
                    counts["complete"] += 1
                elif st == "partial":
                    counts["partial"] += 1
                    partial_reasons[r.get("airr_assembly_reason", "unknown")] += 1
                elif st == "failed":
                    counts["failed"] += 1
                    fail_reasons[r.get("airr_assembly_error", "?").split(":")[0]] += 1
                if r.get("length_cropped"):
                    counts["cropped"] += 1
                if r.get("orientation_id") in (2, 3):   # complement/reverse-only: not AIRR-representable
                    counts["nonstandard_orientation"] += 1
        writer.close(commit=True)
        committed = True
    finally:
        if not committed:
            writer.close(commit=False)              # discard partial output if the pass raised
        if rej_f:
            rej_f.close()
            os.replace(rej_tmp, rejects_out)
    counts["input"] = counts["accepted"] + counts["rejected"]
    return counts, dict(fail_reasons), dict(partial_reasons)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_run_metadata(out: str, model_path: str, args, device: str, counts: dict, *,
                        loci=None, fail_reasons: dict | None = None, partial_reasons: dict | None = None) -> None:
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
                    "locus": args.locus, "batch_size": args.batch_size, "chunk_size": args.chunk_size,
                    "device": device},
        "counts": {                                    # full record accounting (AIRR-review item 5)
            "input_records": counts["input"], "accepted_records": counts["accepted"],
            "rejected_records": counts["rejected"], "cropped_records": counts["cropped"],
            "complete_assemblies": counts["complete"], "partial_assemblies": counts["partial"],
            "failed_assemblies": counts["failed"], "written_records": counts["written"],
            # complement/reverse-only reads: `sequence` is transformed and rev_comp can't express it
            # (see the `orientation` extension). A surprising rate is a useful QC signal.
            "nonstandard_orientation_records": counts["nonstandard_orientation"],
        },
        "airr_assembly_fail_reasons": fail_reasons or {}, "airr_assembly_partial_reasons": partial_reasons or {},
        "offline": bool(args.offline),
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    with open(out + ".run.json", "w") as f:
        json.dump(prov, f, indent=2)


def run(args) -> int:
    import torch
    from ..model_file import container
    from ..registry import maybe_notify_updates, resolve_model, sources as _sources
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.chunk_size <= 0 or args.batch_size <= 0:
        print("--chunk-size and --batch-size must be positive")
        return 1
    for name, val in (("--max-assembly-failures", args.max_assembly_failures),
                      ("--max-partial-assemblies", args.max_partial_assemblies)):
        if not (0.0 <= val <= 1.0):
            print(f"{name} must be a fraction in [0, 1] (got {val})")
            return 1

    # 1) resolve + load the model ONCE (before reading, so a 1M-read input never sits in memory)
    srcs = _sources.resolve_sources(args.registry)
    try:
        model_path = str(resolve_model(args.model, sources=srcs, offline=args.offline,
                                       token=args.hf_token, revision=args.revision))
    except Exception as e:
        print(f"could not resolve model '{args.model}': {e}")
        return 1
    if not container.is_alignair_file(model_path) and not args.dataconfig:
        print("--dataconfig is required for legacy .pt models")
        return 1
    try:                        # the CLI is a thin client of the stable Aligner API (P0-9)
        aligner = Aligner.from_pretrained(model_path, device=device, dataconfigs=args.dataconfig,
                                          trust_pickle=args.trust_pickle)
    except ValueError as e:
        print(str(e))
        return 1
    reference = aligner.reference

    # locus: don't silently default to IGH when the model declares its locus (P0-6)
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

    from ..io.sequence_reader import DuplicateMetadataId
    try:
        meta_by_id, extra_cols = _load_metadata_table(args)  # disk-backed join index (bounded lookup)
    except DuplicateMetadataId as e:
        print(f"metadata error: {e}")
        return 1

    # 2) stream reader-chunk -> predict -> assemble -> join -> write, in bounded memory (~chunk_size)
    from ..io.airr import PredictionCountMismatch
    try:
        counts, fail_reasons, partial_reasons = _stream_predict(
            aligner, input_path=args.input, out_path=args.out, columns=args.columns,
            chunk_size=args.chunk_size, seq_column=args.sequence_column, id_column=args.id_column,
            meta_by_id=meta_by_id, extra_cols=extra_cols, out_locus=out_locus, overrides=overrides,
            batch_size=args.batch_size, rejects_out=args.rejects_out)
    except PredictionCountMismatch as e:                  # a defect returned != 1 prediction per read
        print(f"internal error: {e}. No output written.")
        return 1
    finally:
        if hasattr(meta_by_id, "close"):                 # release the disk-backed metadata index
            meta_by_id.close()

    if not args.no_run_metadata:
        _write_run_metadata(args.out, model_path, args, device, counts, loci=loci,
                            fail_reasons=fail_reasons, partial_reasons=partial_reasons)
    if counts["accepted"] == 0:
        print(f"no valid reads in {args.input} (wrote {counts['rejected']} to rejects)")
        return 1

    # 3) assembly gate on the AGGREGATE rate (output is already written + tagged, like the eager path)
    n = counts["written"]
    fail_rate = counts["failed"] / n if n else 0.0
    partial_rate = counts["partial"] / n if n else 0.0
    if not args.permissive and counts["failed"] and fail_rate > args.max_assembly_failures:
        print(f"AIRR assembly FAILED for {counts['failed']}/{n} reads ({fail_rate:.1%} > "
              f"--max-assembly-failures {args.max_assembly_failures:.1%}); reasons {fail_reasons}. "
              f"Wrote tagged output to {args.out}. Re-run with --permissive or raise the threshold.")
        return 1
    if not args.permissive and counts["partial"] and partial_rate > args.max_partial_assemblies:
        print(f"AIRR assembly PARTIAL for {counts['partial']}/{n} reads ({partial_rate:.1%} > "
              f"--max-partial-assemblies {args.max_partial_assemblies:.1%}); reasons {partial_reasons}. "
              f"Wrote tagged output to {args.out}. Re-run with --permissive or raise the threshold.")
        return 1

    msg = f"aligned {counts['accepted']} reads ({counts['rejected']} dropped) -> {args.out}"
    if counts["failed"] or counts["partial"]:
        msg += f"; {counts['failed']} failed / {counts['partial']} partial AIRR assemblies tagged"
    print(msg)

    # a path, a pinned id/revision, or a direct HF repo: no catalog-update noise
    pinned = (os.path.exists(args.model) or "@" in args.model or bool(args.revision)
              or args.model.startswith("hf://"))
    maybe_notify_updates(sources_list=srcs, offline=args.offline, quiet=args.quiet, pinned=pinned)
    return 0
