"""``alignair predict`` — align reads with a trained model and write an AIRR TSV.

``--model`` accepts a filesystem path OR a shipped registry id / ``id@version`` (resolved + verified
into the local cache). Writes an ``<out>.run.json`` provenance sidecar and passively notes if a newer
model is available (suppressed for a pinned id / a path / ``--offline`` / ``--quiet``).
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

from .. import __version__
from ..aligner import Aligner
from ..errors import (AlignAIRError, ConfigurationError, DataContractError, InputError, ModelError,
                      RegistryError, report_error)
from ..genotype.lean import EXPLAIN, MIN_READS
from ..reference.reference_set import FALLBACK_LOCUS
from ._errors import fail


def register(sub) -> None:
    p = sub.add_parser("predict", help="align reads with a trained model -> AIRR TSV")
    p.add_argument("--model", required=True,
                   help="trained model: a .alignair path, or a shipped registry id / id@version")
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
                        "if it declares exactly one; else IGH). With several loci, each record carries its called "
                        "V's locus (blank without a V call).")
    p.add_argument("--columns", default=None,
                   help="output columns: a preset (full/core/minimal/airr) or a comma-separated "
                        "field list (default: full). Light selections skip the AIRR assembly for speed.")
    p.add_argument("--device", default=None, help="cpu / cuda / mps / auto (default: auto: CUDA, then Apple MPS, then CPU)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="reads per batch (default: 256 on a GPU, 16 per worker with --threads on the CPU, else 64)")
    p.add_argument("--threads", type=int, default=None,
                   help="CPU threads: on the CPU, N worker processes of one thread each (torch's threads scale "
                        "poorly on read batches); on a GPU, torch's CPU threads (default: one process on torch's "
                        "own thread count)")
    p.add_argument("--chunk-size", type=int, default=20000,
                   help="stream the input in chunks of this many reads (default 20000)")
    p.add_argument("--genotype", default=None,
                   help="genotype JSON/YAML {V|D|J: [allele names]}: the bank the calls come from, a subset of the "
                        "bound reference (--reference may add novel alleles)")
    p.add_argument("--reference", default=None,
                   help="germline FASTA (V, D and J alleles) to call from, novel alleles included. Missing IMGT "
                        "gaps and anchors are taken from the model's own reference (the same allele, or a close "
                        "same-length variant)")
    p.add_argument("--infer-genotype", action="store_true",
                   help="infer the donor's V genotype from the reads' own calls (TIgGER's frequency rule on the "
                        "reads whose V call is a single allele) and re-call the reads whose call it excludes while "
                        "they tie an allele it allows; writes the genotype to --genotype-out. Reads the input twice")
    p.add_argument("--genotype-out", default=None,
                   help="where --infer-genotype writes the genotype, an AIRR GenotypeSet (default: <out>.genotype.json)")
    p.add_argument("--genotype-explain", type=float, default=EXPLAIN,
                   help=f"--infer-genotype: the share of a gene's single-allele reads its alleles explain "
                        f"(default {EXPLAIN})")
    p.add_argument("--genotype-min-reads", type=int, default=MIN_READS,
                   help=f"--infer-genotype: the single-allele reads a gene needs to be genotyped; a gene with fewer "
                        f"keeps every allele (default {MIN_READS})")
    p.add_argument("--registry", action="append", help="registry url for a shipped model id (repeatable)")
    p.add_argument("--revision", default=None,
                   help="pin a Hugging Face branch/tag/commit (for hf://org/repo) or a catalog version")
    p.add_argument("--hf-token", default=None,
                   help="Hugging Face token for a private/gated repo (falls back to $HF_TOKEN, then "
                        "$HUGGING_FACE_HUB_TOKEN)")
    p.add_argument("--offline", action="store_true", help="never touch the network")
    p.add_argument("--quiet", action="store_true", help="suppress the update-available notice")
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
    up by original source ID through a temporary disk-backed index.

    Metadata may NEVER overwrite a model/scientific AIRR field: any metadata column colliding with a
    produced field (sequence/v_call/productive/coords/…) is namespaced to ``meta_<col>`` so the aligner's
    result is preserved. 10x column names are normalized to AIRR."""
    if not args.metadata:
        return None, None
    from ..io.airr import COLUMNS, _METADATA_FILL_ONLY
    from ..io.metadata import build_metadata_index
    keep = [c.strip() for c in args.keep_columns.split(",")] if args.keep_columns else None
    protected = frozenset(COLUMNS)

    def _safe(col):                                # protect model fields from metadata clobbering
        if col in _METADATA_FILL_ONLY:             # keep AIRR name (c_call); writer fills only when blank
            return col
        return f"meta_{col}" if col in protected else col
    index, kept_safe = build_metadata_index(args.metadata, id_column=args.metadata_id_column,
                                            keep_columns=keep, normalize_10x=True, rename=_safe)
    return index, kept_safe


def _stream_predict(aligner, *, input_path, out_path, columns, chunk_size, seq_column, id_column,
                    meta_by_id, extra_cols, out_locus, batch_size, rejects_out, pool=None,
                    observe=None):
    """Stream reader-chunk -> predict -> AIRR assembly -> metadata join -> writer -> counters, in
    chunks of sequence/metadata payloads. ID reservations grow with distinct accepted output IDs.
    Order and cross-chunk ID handling are preserved; rejects are written incrementally. Returns
    ``(counts, fail_reasons, partial_reasons)``. The output is committed atomically on a clean pass.
    ``pool`` (a ``predict.parallel.ProcessPool``), when given, predicts each chunk in its workers. ``observe``, when
    given, is called with each chunk's first row index and its records before they are counted."""
    import csv
    from collections import Counter

    from ..io.airr import AirrWriter, needs_assembly
    from ..io.sequence_reader import iter_sequence_batches
    from ..io.metadata import lookup_metadata_rows
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
        with closing(iter_sequence_batches(input_path, chunk_size=chunk_size,
                                            seq_column=seq_column, id_column=id_column,
                                            rejects=rejects_buf)) as batches:
            for batch in batches:
                cids, cseqs, dropped = batch.ids, batch.sequences, batch.n_dropped
                counts["rejected"] += dropped
                if rejects_buf is not None:              # drain this chunk's rejects to disk, keep memory flat
                    for rj in rejects_buf:
                        rej_w.writerow(rj)
                    rejects_buf.clear()
                if not cseqs:
                    continue
                records = (pool.predict(cseqs, batch_size=batch_size, airr=do_airr) if pool is not None
                           else aligner.predict(cseqs, batch_size=batch_size, airr=do_airr).to_dicts())
                if observe is not None:
                    observe(counts["written"], records)
                metas = lookup_metadata_rows(meta_by_id, batch.source_ids)  # Batched lookup by original source ID
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
        if rej_f:                                   # the rejects are published with the output, or not at all
            rej_f.close()
            if committed:
                os.replace(rej_tmp, rejects_out)
            else:
                os.remove(rej_tmp)
    counts["input"] = counts["accepted"] + counts["rejected"]
    return counts, dict(fail_reasons), dict(partial_reasons)


def output_locus(requested: str | None, loci: tuple) -> str:
    """The locus written on a record that carries none, and on an inferred genotype: ``requested`` (``--locus``) as
    the model spells it, which must be one of the model's ``loci`` when it declares any; without it the model's one
    locus, else IGH."""
    if requested is None:
        return loci[0] if len(loci) == 1 else FALLBACK_LOCUS
    spelled = {locus.upper(): locus for locus in loci}
    if spelled and requested.upper() not in spelled:
        raise ConfigurationError(f"--locus {requested} is not one of this model's loci {loci}",
                                 context={"locus": requested})
    return spelled.get(requested.upper(), requested.upper())


def _load_reporting_locus(model_path: str, *, device: str, reference):
    """``Aligner.from_pretrained``, its ``LocusMismatchWarning`` (a reference of another locus bound to a one-locus
    model) printed as one ``warning:`` line on stderr; any other warning passes through as it would."""
    import warnings
    from ..reference import LocusMismatchWarning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        aligner = Aligner.from_pretrained(model_path, device=device, reference=reference)
    for w in caught:
        if issubclass(w.category, LocusMismatchWarning):
            print(f"warning: {w.message}", file=sys.stderr)
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno, source=w.source)
    return aligner


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_run_metadata(out: str, model_path: str, args, device: str, counts: dict, *,
                        loci=None, fail_reasons: dict | None = None, partial_reasons: dict | None = None,
                        genotype_pass: dict | None = None, bound_reference=None, genotype_bank=None) -> None:
    """The run's provenance sidecar. ``reference_fasta_sha256`` / ``allele_order_sha256`` fingerprint the reference
    the calls were made from: ``bound_reference`` (the ``--reference`` FASTA as annotated and bound) when one was
    given, else the model's embedded one; ``reference`` names the source and keeps the embedded fingerprints, and
    ``genotype`` the ``--genotype`` file and its bank's allele order."""
    from ..model_file import container, read_metadata, serialize
    md = read_metadata(model_path) if container.is_alignair_file(model_path) else {}
    ref = md.get("reference", {})
    given = args.reference is not None and bound_reference is not None
    prov = {
        "model_spec": args.model, "model_path": os.path.abspath(model_path),
        "model_id": md.get("model_id"), "model_version": md.get("model_version"),
        "artifact_sha256": _sha256_file(model_path),
        "reference_fasta_sha256": (serialize.reference_fasta_sha256(bound_reference) if given
                                   else ref.get("reference_fasta_sha256")),
        "allele_order_sha256": (serialize.allele_order_sha256(bound_reference) if given
                                else ref.get("allele_order_sha256")),
        "reference": {"source": "--reference" if given else "embedded",
                      "path": args.reference if given else None,
                      "file_sha256": _sha256_file(args.reference) if given else None,
                      "embedded_reference_fasta_sha256": ref.get("reference_fasta_sha256"),
                      "embedded_allele_order_sha256": ref.get("allele_order_sha256")},
        "genotype": ({"path": args.genotype, "file_sha256": _sha256_file(args.genotype),
                      "allele_order_sha256": serialize.allele_order_sha256(genotype_bank)}
                     if args.genotype and genotype_bank is not None else None),
        "alignair_version": __version__, "model_created_by_alignair": md.get("created_by_alignair"),
        "loci": list(loci) if loci else [],  # the bound reference's locus mapping
        "command": {"input": args.input, "out": args.out, "columns": args.columns,
                    "locus": args.locus, "batch_size": args.batch_size, "chunk_size": args.chunk_size, "threads": args.threads,
                    "device": device, "reference": args.reference, "genotype": args.genotype},
        "counts": {  # full record accounting
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
        **({"genotype_pass": genotype_pass} if genotype_pass is not None else {}),
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    with open(out + ".run.json", "w") as f:
        json.dump(prov, f, indent=2)


def run(args) -> int:
    import torch
    from ..model_file import container
    from ..registry import maybe_notify_updates, resolve_model, sources as _sources
    from ..aligner import resolve_device
    device = resolve_device(args.device or "auto")     # as the API's device="auto": CUDA, then MPS, then CPU

    if args.chunk_size <= 0 or (args.batch_size is not None and args.batch_size <= 0):
        return report_error(ConfigurationError("--chunk-size and --batch-size must be positive"))
    if args.threads is not None and args.threads < 1:
        return report_error(ConfigurationError(f"--threads must be at least 1 (got {args.threads})"))
    if device == "cpu" and args.threads is not None and args.threads > 1:
        from ..predict import parallel             # the workers' forkserver imports while the model loads
        parallel.prestart()
    for name, val in (("--max-assembly-failures", args.max_assembly_failures),
                      ("--max-partial-assemblies", args.max_partial_assemblies)):
        if not (0.0 <= val <= 1.0):
            return report_error(ConfigurationError(f"{name} must be a fraction in [0, 1] (got {val})"))
    if args.infer_genotype:
        if args.genotype:
            return report_error(ConfigurationError("--infer-genotype infers the genotype; it cannot take --genotype"))
        if "-" in (args.input, args.out):
            return report_error(ConfigurationError(
                "--infer-genotype reads the input twice and rewrites the output: give files, not '-'"))
        if not (0.0 < args.genotype_explain <= 1.0) or args.genotype_min_reads < 1:
            return report_error(ConfigurationError("--genotype-explain must be in (0, 1] and --genotype-min-reads "
                                                   "at least 1"))

    # 1) resolve + load the model ONCE (before reading, so a 1M-read input never sits in memory)
    srcs = _sources.resolve_sources(args.registry)
    try:
        model_path = str(resolve_model(args.model, sources=srcs, offline=args.offline,
                                       token=args.hf_token, revision=args.revision))
    except (AlignAIRError, OSError, ValueError) as e:
        return fail(e, error_type=RegistryError, message=f"could not resolve model '{args.model}'",
                    context={"model_id": args.model})
    if not container.is_alignair_file(model_path):      # an AlignAIR 3 checkpoint: refused with the version that runs it
        from ..model_file import v3_checkpoint_error
        return report_error(v3_checkpoint_error(model_path))
    bound = None
    if args.reference:      # the FASTA's alleles become the model's bank, annotated against its own reference
        from ..reference.reference_set import ReferenceSet
        try:
            bound = ReferenceSet.from_fasta(args.reference)
        except (ValueError, OSError) as e:
            return fail(e, context={"stage": "reference_binding", "path": args.reference})
    try:  # the CLI is a thin client of the stable Aligner API
        aligner = _load_reporting_locus(model_path, device=device, reference=bound)
    except ValueError as e:                   # a ModelError is the file's; any other error with a reference, binding's
        if bound is None or isinstance(e, ModelError):
            return fail(e, error_type=ModelError, context={"path": model_path})
        return fail(e, context={"stage": "reference_binding", "path": args.reference})
    except DataContractError as e:
        if bound is None:
            raise
        return fail(e, context={"stage": "reference_binding", "path": args.reference})
    if bound is not None:
        from ..reference import anchor_report
        report = anchor_report(aligner.reference)
        if not report.complete() or report.noncanonical:
            print(f"warning: --reference {args.reference}: {report.summary()}", file=sys.stderr)
    reference = aligner.reference
    bank = None

    loci = reference.locus_names()
    out_locus = output_locus(args.locus, loci)

    if args.genotype:
        from ..genotype.bank import genotype_bank
        from ..genotype.files import load_genotype
        try:                    # an allele the bound reference lacks is refused, not dropped
            genotype = load_genotype(args.genotype)
            bank = genotype_bank(reference, genotype)
            aligner = aligner.bound_to(bank)                                    # the genotype is the bank: bound once
        except ValueError as e:
            return fail(e, context={"stage": "genotype_validation"})

    meta_by_id, extra_cols = _load_metadata_table(args)  # disk-backed join index (bounded lookup)

    # threads: on the CPU, worker processes of one thread each (predict.parallel); elsewhere torch's CPU threads
    pool = None
    if args.threads is not None:
        if device == "cpu" and args.threads > 1:
            from ..predict.parallel import ProcessPool
            pool = ProcessPool(aligner, args.threads)
        else:
            torch.set_num_threads(args.threads)
    from ..aligner import _auto_batch_size
    args.batch_size = args.batch_size or _auto_batch_size(device, pool.workers if pool is not None else 1)

    # 2) stream reader-chunk -> predict -> assemble -> join -> write, in bounded memory (~chunk_size)
    from ..io.airr import needs_assembly
    from . import genotype_pass as gp
    observer = gp.Observer() if args.infer_genotype else None
    first = f"{args.out}.pass1.{os.getpid()}" if observer is not None else args.out
    genotype_record = None
    try:
        counts, fail_reasons, partial_reasons = _stream_predict(
            aligner, input_path=args.input, out_path=first, columns=args.columns,
            chunk_size=args.chunk_size, seq_column=args.sequence_column, id_column=args.id_column,
            meta_by_id=meta_by_id, extra_cols=extra_cols, out_locus=out_locus,
            batch_size=args.batch_size, rejects_out=args.rejects_out, pool=pool, observe=observer)
        if observer is not None:      # infer the genotype, re-call the reads it moves, swap their rows in
            do_airr = needs_assembly(args.columns)

            def bind(genotype):
                """Pass 2's ``predict``: the aligner bound once to ``genotype``'s bank, its workers restarted on it."""
                nonlocal pool
                from ..genotype.bank import genotype_bank
                bound = aligner.bound_to(genotype_bank(reference, genotype))
                if pool is None:
                    return lambda sequences: bound.predict(sequences, batch_size=args.batch_size,
                                                           airr=do_airr).to_dicts()
                from ..predict.parallel import ProcessPool
                pool.close()
                pool = None
                pool = workers = ProcessPool(bound, args.threads)
                return lambda sequences: workers.predict(sequences, batch_size=args.batch_size, airr=do_airr)
            genotype_record = gp.complete(observer, args, first=first, bind=bind,
                                          v_names=reference.gene("V").names, extra_cols=extra_cols,
                                          out_locus=out_locus, meta_by_id=meta_by_id, counts=counts,
                                          fail_reasons=fail_reasons, partial_reasons=partial_reasons)
    finally:
        if observer is not None and os.path.exists(first):
            os.remove(first)
        if pool is not None:
            pool.close()
        if hasattr(meta_by_id, "close"):                 # release the disk-backed metadata index
            meta_by_id.close()

    if not args.no_run_metadata:
        _write_run_metadata(args.out, model_path, args, device, counts, loci=loci,
                            fail_reasons=fail_reasons, partial_reasons=partial_reasons,
                            genotype_pass=genotype_record, bound_reference=reference, genotype_bank=bank)
    if counts["accepted"] == 0:
        return report_error(InputError(f"no valid reads in {args.input} (wrote {counts['rejected']} to rejects)",
                                       context={"path": args.input, "rejected_count": counts["rejected"]}))

    # 3) assembly gate on the AGGREGATE rate (output is already written + tagged, like the eager path)
    n = counts["written"]
    fail_rate = counts["failed"] / n if n else 0.0
    partial_rate = counts["partial"] / n if n else 0.0
    if not args.permissive and counts["failed"] and fail_rate > args.max_assembly_failures:
        return report_error(InputError(f"AIRR assembly FAILED for {counts['failed']}/{n} reads ({fail_rate:.1%} > "
              f"--max-assembly-failures {args.max_assembly_failures:.1%}); reasons {fail_reasons}. "
              f"Wrote tagged output to {args.out}.",
              context={"stage": "assembly_quality_gate", "path": args.out, "reasons": fail_reasons}))
    if not args.permissive and counts["partial"] and partial_rate > args.max_partial_assemblies:
        return report_error(InputError(f"AIRR assembly PARTIAL for {counts['partial']}/{n} reads ({partial_rate:.1%} > "
              f"--max-partial-assemblies {args.max_partial_assemblies:.1%}); reasons {partial_reasons}. "
              f"Wrote tagged output to {args.out}.",
              context={"stage": "assembly_quality_gate", "path": args.out, "reasons": partial_reasons}))

    msg = f"aligned {counts['accepted']} reads ({counts['rejected']} dropped) -> {args.out}"
    if counts["failed"] or counts["partial"]:
        msg += f"; {counts['failed']} failed / {counts['partial']} partial AIRR assemblies tagged"
    print(msg)

    # a path, a pinned id/revision, or a direct HF repo: no catalog-update noise
    pinned = (os.path.exists(args.model) or "@" in args.model or bool(args.revision)
              or args.model.startswith("hf://"))
    maybe_notify_updates(sources_list=srcs, offline=args.offline, quiet=args.quiet, pinned=pinned)
    return 0
