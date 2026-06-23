# PYTHON_ARGCOMPLETE_OK
"""`alignair` command-line interface.

    alignair predict reads.fastq -o rearrangement.tsv --model model.pt
    alignair predict reads.fastq -o out.tsv --model model.pt --genotype donor.yaml
    alignair predict reads.fastq -o out.tsv --model model.pt --genotype donor.fasta

A `--genotype` file (YAML or FASTA) simply becomes the reference for the run, so it
transparently supports both an allele SUBSET and NOVEL alleles (the dynamic-genotype
property) with no extra flags.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("AlignAIR")
    except Exception:
        return "unknown (not pip-installed)"


def cmd_predict(args) -> None:
    import torch
    from .api import load_model
    from .hub import resolve_model

    def log(msg):
        if not args.quiet:
            print(msg, file=sys.stderr, flush=True)      # progress -> stderr (keeps stdout clean)

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = resolve_model(args.model)             # local path / catalog id / HF repo id
    loaded = load_model(model_path, device)
    log(f"device: {device}  |  model: {args.model}")
    try:
        rs, ref_desc, locus = _resolve_run_reference(
            loaded, genotype=args.genotype, dataconfig=args.dataconfig,
            force_locus_mismatch=args.force_locus_mismatch, locus_arg=args.locus, log=log)
        calibration = _resolve_calibration(loaded, args.calibration)
        summary = _run_prediction(
            loaded, rs, ref_desc=ref_desc, locus=locus, calibration=calibration, args=args,
            input_path=args.input, output_path=args.output,
            metadata_path=args.metadata, metadata_id=args.metadata_id,
            model_path=model_path, device=device, log=log)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"error: {e}")
    if summary["n_aligned"] == 0:
        raise SystemExit("error: no valid sequences to align")


def _resolve_run_reference(loaded, *, genotype, dataconfig, force_locus_mismatch, locus_arg, log):
    """Resolve a run's germline reference (--genotype > bundle default > a GenAIRR DataConfig) and
    its locus, enforcing model/reference locus compatibility (#3). Raises ValueError on a problem so
    callers can decide (predict -> abort; batch -> record the sample as failed and continue)."""
    import GenAIRR.data as gdata
    from .reference.reference_set import ReferenceSet
    if genotype:
        if not os.path.exists(genotype):
            raise ValueError(f"genotype file not found: {genotype}")
        ext = os.path.splitext(genotype)[1].lower()
        loader = ReferenceSet.from_fasta if ext in (".fasta", ".fa", ".fna", ".faa") else ReferenceSet.from_yaml
        rs = loader(genotype)
        ref_desc = f"genotype:{os.path.basename(genotype)}"
        log(f"reference: genotype {genotype} (V={len(rs.gene('V').names)}"
            f"{', D=' + str(len(rs.gene('D').names)) if rs.has_d else ''}, J={len(rs.gene('J').names)})")
    elif loaded.reference_set is not None:
        rs = loaded.reference_set
        ref_desc = ", ".join(loaded.dataconfigs) if loaded.dataconfigs else f"bundled ({len(rs.gene('V').names)} V)"
        log(f"reference: {ref_desc} (V={len(rs.gene('V').names)}"
            f"{', D=' + str(len(rs.gene('D').names)) if rs.has_d else ''}, J={len(rs.gene('J').names)})")
    else:
        names = [dataconfig]
        try:
            rs = ReferenceSet.from_dataconfigs(*[getattr(gdata, n) for n in names])
        except AttributeError as e:
            raise ValueError(f"unknown GenAIRR DataConfig in {names}: {e}")
        ref_desc = ", ".join(names)
        log(f"reference: {', '.join(names)} (V={len(rs.gene('V').names)})")

    inferred_locus = rs.infer_locus()
    if inferred_locus and loaded.locus and inferred_locus != loaded.locus:
        msg = (f"model is for locus {loaded.locus} but the reference looks like {inferred_locus} — "
               f"calls would be biologically meaningless. Use a matching model/reference, or pass "
               f"--force-locus-mismatch to override.")
        if force_locus_mismatch:
            log(f"WARNING: {msg}")
        else:
            raise ValueError(msg)
    locus = locus_arg or inferred_locus or loaded.locus or "IGH"
    return rs, ref_desc, locus


def _resolve_calibration(loaded, calibration_path):
    cal = loaded.calibration
    if calibration_path and os.path.exists(calibration_path):
        cal = json.load(open(calibration_path))                   # explicit flag overrides bundle
    return cal


def _run_prediction(loaded, rs, *, ref_desc, locus, calibration, args, input_path, output_path,
                    metadata_path, metadata_id, model_path, device, log, desc="aligning"):
    """Stream one input -> one AIRR TSV (bounded memory) and return a summary dict. Raises ValueError
    on a write/read problem (no exit here, so callers control the failure policy)."""
    from .api import predict as api_predict
    from .io.sequence_reader import iter_sequences, load_metadata
    from .io.airr import AirrWriter
    to_stdout = output_path == "-"
    write_provenance = not (args.no_provenance or to_stdout)
    chunk = max(args.batch, args.chunk_size)
    # optional per-read metadata join (10x annotations / AIRR metadata) -> preserved in output
    meta_map, extra_cols = {}, []
    if metadata_path:
        kc = [c.strip() for c in args.keep_columns.split(",")] if args.keep_columns else None
        meta_map, extra_cols = load_metadata(metadata_path, metadata_id, kc)  # raises on bad columns/file
        log(f"metadata: {len(meta_map)} rows from {metadata_path}; preserving columns {extra_cols}")
    bar = None if (args.quiet or to_stdout) else __import__("tqdm").tqdm(
        desc=desc, unit="read", file=sys.stderr)
    total = n_prod = n_contam = n_dropped = n_meta_hit = 0
    try:
        if not to_stdout:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        writer = AirrWriter(output_path, locus, extra_columns=extra_cols)
    except (PermissionError, OSError) as e:
        raise ValueError(
            f"cannot write output to {output_path}: {e}\n"
            f"hint: if running in Docker, mount a writable output directory and add "
            f"`--user $(id -u):$(id -g)` so files are written as you.")
    try:
        chunks = iter_sequences(input_path, chunk_size=chunk,
                                seq_column=args.sequence_column, id_column=args.id_column)
        for ids, seqs, drp in chunks:
            n_dropped += drp
            if not seqs:
                continue
            batch = api_predict(loaded, seqs, reference=rs, batch_size=args.batch,
                                v_reader=args.v_reader, calibration=calibration,
                                full_alignment=not args.no_full_alignment, locus=locus)
            canon, preds = batch.sequences, batch.predictions    # canonical (forward) seqs
            metas = None
            if meta_map:
                metas = [meta_map.get(sid, {}) for sid in ids]
                n_meta_hit += sum(1 for m in metas if m)
            writer.write(ids, canon, preds, metas=metas)
            total += len(preds)
            n_prod += sum(1 for p in preds if p.get("productive"))
            n_contam += sum(1 for p in preds if p.get("is_contaminant"))
            if bar is not None:
                bar.update(len(preds))
    finally:
        writer.close()
        if bar is not None:
            bar.close()
    summary = {"output": output_path, "n_aligned": total, "n_productive": n_prod,
               "n_contaminant": n_contam, "n_dropped": n_dropped, "n_meta_hit": n_meta_hit,
               "locus": locus, "metadata": bool(meta_map)}
    if total == 0:
        return summary                       # caller decides whether 0 reads is an error
    if write_provenance:
        from .io.sequence_reader import _detect_format
        info = {"n_read": total + n_dropped, "n_dropped": n_dropped,
                "format": _detect_format(input_path)}
        _write_provenance(output_path + ".run.json", args=args, model_path=model_path,
                          device=device, info=info, ref_desc=ref_desc, output=output_path,
                          reference_set=rs, calibration=calibration)
    if not to_stdout:
        log(f"wrote {total} rearrangements ({n_dropped} dropped) -> {output_path}"
            + (f"  (+ {os.path.basename(output_path)}.run.json)" if write_provenance else ""))
        log(f"summary: {total} aligned | {n_prod} productive ({n_prod/max(total,1)*100:.0f}%)"
            f"{f' | {n_contam} flagged out-of-scope' if n_contam else ''}"
            f"{f' | metadata matched {n_meta_hit}/{total}' if meta_map else ''} | locus {locus}")
    return summary


def cmd_batch(args) -> None:
    """Align many samples with ONE model load. A manifest (CSV/TSV) of `sample_id,input` (optional
    `genotype,metadata` per row) drives one AIRR TSV per sample under -o, plus a manifest_summary
    .tsv/.json with per-sample stats — easier to wrap (Nextflow/Snakemake) and audit than a shell loop."""
    import time
    import torch
    from .api import load_model
    from .hub import resolve_model

    def log(msg):
        if not args.quiet:
            print(msg, file=sys.stderr, flush=True)

    rows = _read_manifest(args.manifest)
    if not rows:
        raise SystemExit(f"error: manifest {args.manifest} has no usable rows "
                         f"(need columns: sample_id, input)")
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = resolve_model(args.model)
    loaded = load_model(model_path, device)              # loaded ONCE, reused for every sample
    log(f"device: {device}  |  model: {args.model}  |  {len(rows)} samples")
    os.makedirs(args.output, exist_ok=True)

    ref_cache, summaries = {}, []
    for i, row in enumerate(rows, 1):
        sid = row["sample_id"]
        out_tsv = os.path.join(args.output, f"{sid}.tsv")
        genotype = row.get("genotype") or args.genotype
        t0 = time.monotonic()
        log(f"[{i}/{len(rows)}] {sid}: {row['input']} -> {out_tsv}")
        try:
            key = genotype or "<model-default>"
            if key not in ref_cache:                     # a shared genotype is resolved once
                ref_cache[key] = _resolve_run_reference(
                    loaded, genotype=genotype, dataconfig=args.dataconfig,
                    force_locus_mismatch=args.force_locus_mismatch, locus_arg=args.locus, log=log)
            rs, ref_desc, locus = ref_cache[key]
            calibration = _resolve_calibration(loaded, args.calibration)
            s = _run_prediction(
                loaded, rs, ref_desc=ref_desc, locus=locus, calibration=calibration, args=args,
                input_path=row["input"], output_path=out_tsv,
                metadata_path=row.get("metadata") or None, metadata_id=args.metadata_id,
                model_path=model_path, device=device, log=log, desc=sid)
            s.update(sample_id=sid, input=row["input"], seconds=round(time.monotonic() - t0, 2),
                     status="ok" if s["n_aligned"] else "empty", error="")
        except (ValueError, FileNotFoundError) as e:
            log(f"  ! {sid} failed: {str(e).splitlines()[0]}")
            s = {"sample_id": sid, "input": row["input"], "output": out_tsv, "status": "error",
                 "error": str(e).splitlines()[0], "n_aligned": 0, "n_productive": 0,
                 "n_contaminant": 0, "n_dropped": 0, "locus": "",
                 "seconds": round(time.monotonic() - t0, 2)}
        summaries.append(s)

    _write_manifest_summary(args.output, summaries)
    ok = sum(1 for s in summaries if s["status"] == "ok")
    empty = sum(1 for s in summaries if s["status"] == "empty")
    failed = [s["sample_id"] for s in summaries if s["status"] == "error"]
    log(f"done: {ok} ok"
        + (f", {empty} empty" if empty else "")
        + (f", {len(failed)} failed" if failed else "")
        + f" / {len(summaries)} samples -> {args.output}/  (manifest_summary.tsv)")
    if failed:
        log(f"failed samples: {', '.join(failed)}")
    if ok == 0:                                          # total failure -> non-zero exit for wrappers
        raise SystemExit("error: no samples aligned successfully")


def _read_manifest(path):
    """Parse a sample manifest (CSV/TSV). Required columns: sample_id, input. Optional: genotype,
    metadata. Relative paths are tried as-is, then relative to the manifest's directory."""
    if not os.path.exists(path):
        raise SystemExit(f"error: manifest not found: {path}")
    import csv
    base = os.path.dirname(os.path.abspath(path))

    def resolve(p):
        if not p or os.path.isabs(p) or os.path.exists(p):
            return p
        cand = os.path.join(base, p)
        return cand if os.path.exists(cand) else p

    with open(path, newline="") as f:
        delim = "\t" if "\t" in f.readline() else ","
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)
        cols = {(c or "").strip() for c in (reader.fieldnames or [])}
        missing = {"sample_id", "input"} - cols
        if missing:
            raise SystemExit(f"error: manifest {path} must have columns sample_id,input "
                             f"(missing {sorted(missing)}); optional: genotype, metadata")
        rows = []
        for r in reader:
            r = {(k or "").strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
            if not r.get("sample_id") or not r.get("input"):
                continue
            for col in ("input", "genotype", "metadata"):
                if r.get(col):
                    r[col] = resolve(r[col])
            rows.append(r)
    return rows


def _write_manifest_summary(out_dir, summaries):
    import csv
    cols = ["sample_id", "status", "n_aligned", "n_productive", "n_contaminant", "n_dropped",
            "locus", "seconds", "output", "input", "error"]
    with open(os.path.join(out_dir, "manifest_summary.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for s in summaries:
            w.writerow(s)
    with open(os.path.join(out_dir, "manifest_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)


def _pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _write_provenance(path, *, args, model_path, device, info, ref_desc, output=None,
                      reference_set=None, calibration=None):
    """Write a run.json sidecar next to the output: enough to reproduce/audit this file (the AIRR
    Software WG expects run parameters to travel with the output). Includes the AlignAIR source
    commit, package versions, CUDA detail, and reference/calibration content hashes."""
    import datetime
    from .provenance import (alignair_version, package_versions, git_commit_sha, cuda_detail,
                             hash_json, reference_hash)
    fingerprint = None
    fp = os.path.join(model_path, "fingerprint.txt") if os.path.isdir(model_path) else None
    if fp and os.path.exists(fp):
        fingerprint = open(fp).read().strip()
    model_build = None                                   # carry the bundle's training provenance
    mp = os.path.join(model_path, "meta.json") if os.path.isdir(model_path) else None
    if mp and os.path.exists(mp):
        try:
            m = json.loads(open(mp).read())
            model_build = {k: m.get(k) for k in ("alignair_version", "git_commit", "created_utc")}
        except Exception:
            model_build = None
    prov = {
        "tool": "AlignAIR", "alignair_version": alignair_version(),
        "git_commit": git_commit_sha(),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": "alignair " + " ".join(sys.argv[1:]),
        "model": model_path, "model_fingerprint": fingerprint, "model_build": model_build,
        "reference": ref_desc, "reference_hash": reference_hash(reference_set),
        "calibration_hash": hash_json(calibration),
        "device": device, "cuda": cuda_detail(),
        "v_reader": args.v_reader, "batch": args.batch, "seed": args.seed,
        "n_input": info.get("n_read"), "n_dropped": info.get("n_dropped"),
        "input_format": info.get("format"), "output": output or getattr(args, "output", None),
        "versions": package_versions(),
    }
    with open(path, "w") as f:
        json.dump(prov, f, indent=2)


def cmd_compare(args) -> None:
    """Compare two AIRR rearrangement TSVs (e.g. AlignAIR vs IgBLAST/MiXCR) on the SAME reads —
    agreement, disagreements, and AlignAIR's equivalence-set rescue. No ground truth needed."""
    from .compare import read_airr, compare_airr, format_report_md
    for f in (args.a, args.b):
        if not os.path.exists(f):
            raise SystemExit(f"error: file not found: {f}")
    a, b = read_airr(args.a), read_airr(args.b)
    if not (set(a) & set(b)):
        raise SystemExit("error: no shared sequence_id between the two files — are both AIRR TSVs "
                         "from the same reads?")
    report = compare_airr(a, b, a_name=args.a_name, b_name=args.b_name)
    md = format_report_md(report)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(md + "\n")
        if args.json:
            json.dump(report, open(args.json, "w"), indent=2)
        print(f"wrote agreement report -> {args.out}" + (f" (+ {args.json})" if args.json else ""))
    else:
        print(md)
        if args.json:
            json.dump(report, open(args.json, "w"), indent=2)


def cmd_validate_airr(args) -> None:
    """Validate a rearrangement TSV against the official AIRR-C schema (needs the `airr` package)."""
    try:
        import airr
    except ImportError:
        raise SystemExit("error: AIRR validation needs the `airr` package "
                         "— install with `pip install \"AlignAIR[cli]\"` or `pip install airr`")
    if not os.path.exists(args.file):
        raise SystemExit(f"error: file not found: {args.file}")
    try:
        ok = bool(airr.validate_rearrangement(args.file))
    except Exception as e:
        raise SystemExit(f"AIRR validation error: {e}")
    print(f"{args.file}: {'VALID AIRR-C rearrangement' if ok else 'NOT valid AIRR-C'}")
    raise SystemExit(0 if ok else 1)


def cmd_model(args) -> None:
    """Manage pretrained models: list the catalog, download a bundle, or inspect one."""
    from .hub import MODEL_CATALOG, resolve_model
    from .serialization.dnalignair_bundle import is_bundle, load_dnalignair_bundle
    if args.model_command == "list":
        if getattr(args, "json", False):
            print(json.dumps(MODEL_CATALOG, indent=2))
            return
        if not MODEL_CATALOG:
            print("(no models in the catalog yet)")
            return
        avail = {k: e for k, e in MODEL_CATALOG.items() if e.get("available", True)}
        planned = {k: e for k, e in MODEL_CATALOG.items() if not e.get("available", True)}
        if avail:
            print("AVAILABLE:")
            print(f"  {'id':22s} {'species':8s} {'locus':6s} description")
            for mid, e in avail.items():
                print(f"  {mid:22s} {e.get('species',''):8s} {e.get('locus',''):6s} {e.get('description','')}")
            print("  download: alignair model download <id>   |   use: alignair predict ... --model <id>")
        else:
            print("AVAILABLE: (none published yet)")
        if planned:
            print("\nPLANNED (not downloadable — train your own with `alignair train`, or `alignair demo`):")
            for mid, e in planned.items():
                print(f"  {mid:22s} {e.get('species',''):8s} {e.get('locus',''):6s} {e.get('description','')}")
    elif args.model_command == "download":
        path = resolve_model(args.id, dest=args.dest)
        print(f"downloaded -> {path}")
    elif args.model_command == "inspect":
        path = resolve_model(args.id)
        if not is_bundle(path):
            raise SystemExit(f"error: {path} is not an AlignAIR bundle (raw checkpoints carry no metadata)")
        b = load_dnalignair_bundle(path, build=False)
        cfg = b["config"].to_dict()
        meta = b["meta"] or {}
        embedded = bool(b.get("reference_set"))
        info = {"bundle": path, "locus": b["locus"],
                "reference": {"embedded": embedded,
                              "v_alleles": len(b["reference_set"].gene("V")) if embedded else None,
                              "dataconfigs": None if embedded else b["dataconfigs"]},
                "config": {k: cfg.get(k) for k in ("d_model", "n_layers", "nhead")},
                "calibration": bool(b["calibration"]),
                "notes": meta.get("notes"),
                "provenance": {k: meta.get(k) for k in (
                    "alignair_version", "git_commit", "created_utc", "versions",
                    "reference_hash", "config_hash", "calibration_hash", "training")}}
        if getattr(args, "json", False):
            print(json.dumps(info, indent=2))
            return
        ref = (f"{info['reference']['v_alleles']} V (embedded)" if embedded
               else f"dataconfigs={b['dataconfigs']}")
        prov = info["provenance"]
        print(f"bundle: {path}")
        print(f"  locus       : {b['locus']}")
        print(f"  reference   : {ref}")
        print(f"  d_model={cfg.get('d_model')} n_layers={cfg.get('n_layers')} nhead={cfg.get('nhead')}")
        print(f"  calibration : {'yes' if b['calibration'] else 'no'}")
        print(f"  built with  : AlignAIR {prov.get('alignair_version')}"
              f"{' @ ' + prov['git_commit'] if prov.get('git_commit') else ''}"
              f"{' on ' + prov['created_utc'] if prov.get('created_utc') else ''}")
        print(f"  reference_hash : {prov.get('reference_hash')}")
        if prov.get("training"):
            print(f"  training    : {prov['training']}")
        print(f"  notes       : {meta.get('notes')}")


def _load_reference_file(path):
    from .reference.reference_set import ReferenceSet
    if not os.path.exists(path):
        raise SystemExit(f"error: file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    return (ReferenceSet.from_fasta(path) if ext in (".fasta", ".fa", ".fna", ".faa")
            else ReferenceSet.from_yaml(path))


def cmd_reference(args) -> None:
    """Validate or convert a germline reference (YAML genotype <-> FASTA)."""
    if args.reference_command == "list":
        names = _genairr_dataconfigs()
        if getattr(args, "json", False):
            print(json.dumps({"dataconfigs": names, "chain_types": _genairr_chain_types()}, indent=2))
            return
        print(f"{len(names)} built-in GenAIRR references (use as `--reference NAME`):\n")
        print(_format_dataconfig_list())
        print("\nvalid `--chain-type` for custom FASTA references "
              "(--v-fasta/--j-fasta[/--d-fasta]):")
        print("  " + ", ".join(_genairr_chain_types()))
        return
    if args.reference_command == "validate":
        rs = _load_reference_file(args.file)
        ok = True
        print(f"reference: {args.file}")
        for G in ("V", "D", "J"):
            if G not in rs.genes:
                continue
            ref = rs.gene(G)
            n = len(ref.names)
            dups = n - len(set(ref.names))
            empties = sum(1 for s in ref.sequences if not s)
            nonacgt = sum(1 for s in ref.sequences if set(s.upper()) - set("ACGTN"))
            anc = len(ref.anchors or {})
            flags = []
            if dups: flags.append(f"{dups} DUPLICATE names"); ok = False
            if empties: flags.append(f"{empties} EMPTY sequences"); ok = False
            if nonacgt: flags.append(f"{nonacgt} with non-ACGTN chars")
            anc_note = f"{anc}/{n} anchors" + ("" if anc == n or G == "D" else "  (missing anchors need --allow-curatable to train)")
            print(f"  {G}: {n} alleles | {anc_note}" + (("  ⚠ " + "; ".join(flags)) if flags else ""))
        print("status: OK" if ok else "status: PROBLEMS FOUND")
        raise SystemExit(0 if ok else 1)
    elif args.reference_command == "template":
        import GenAIRR.data as gdata
        from .reference.reference_set import ReferenceSet
        if not hasattr(gdata, args.reference):
            raise SystemExit(f"error: unknown GenAIRR DataConfig '{args.reference}'")
        rs = ReferenceSet.from_dataconfigs(getattr(gdata, args.reference))
        rs.to_yaml(args.out)
        n = sum(len(rs.gene(g)) for g in rs.genes)
        print(f"wrote genotype template ({n} alleles from {args.reference}) -> {args.out}")
        print("edit it down to a donor's alleles and/or add NOVEL alleles, then:")
        print(f"  alignair predict reads.fasta -o out.tsv --model <bundle> --genotype {args.out}")
    elif args.reference_command == "convert":
        rs = _load_reference_file(args.file)
        out_ext = os.path.splitext(args.out)[1].lower()
        if out_ext in (".fasta", ".fa", ".fna", ".faa"):
            with open(args.out, "w") as fh:
                for G in ("V", "D", "J"):
                    if G not in rs.genes:
                        continue
                    ref = rs.gene(G)
                    for nm, seq in zip(ref.names, ref.sequences):
                        fh.write(f">{nm}\n{seq}\n")
        else:
            rs.to_yaml(args.out)
        n = sum(len(rs.gene(g)) for g in rs.genes)
        print(f"converted {args.file} -> {args.out} ({n} alleles)")


def cmd_doctor(args) -> None:
    """Environment / install check: Python, PyTorch + CUDA, GenAIRR, optional parasail/airr, and
    (optionally) whether a --model path resolves. Exit non-zero if a CORE dependency is missing.
    `--json` emits a machine-readable report; `--verbose` adds platform/build detail."""
    rep = {"alignair_version": _version(), "ok": True,
           "python": {"version": sys.version.split()[0], "executable": sys.executable},
           "core": {}, "optional": {}}
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        rep["core"]["torch"] = {"present": True, "version": torch.__version__, "cuda": cuda,
                                "device": torch.cuda.get_device_name(0) if cuda else "cpu only"}
    except Exception as e:
        rep["ok"] = False
        rep["core"]["torch"] = {"present": False, "error": str(e)}
    try:
        import GenAIRR
        rep["core"]["GenAIRR"] = {"present": True, "version": getattr(GenAIRR, "__version__", "?")}
    except Exception as e:
        rep["ok"] = False
        rep["core"]["GenAIRR"] = {"present": False, "error": str(e)}
    for name, dist in (("parasail", "parasail"), ("airr", "airr"),
                       ("huggingface_hub", "huggingface-hub"), ("argcomplete", "argcomplete")):
        try:
            __import__(name)
            rep["optional"][name] = {"present": True, "version": _pkg_version(dist)}
        except Exception:
            rep["optional"][name] = {"present": False, "version": None}
    if args.model:
        from .serialization.dnalignair_bundle import is_bundle
        if not os.path.exists(args.model):
            rep["ok"] = False
            rep["model"] = {"path": args.model, "status": "not found"}
        else:
            rep["model"] = {"path": args.model,
                            "status": "bundle" if is_bundle(args.model) else "raw checkpoint"}
    if args.verbose:
        import platform
        rep["platform"] = {"system": platform.system(), "release": platform.release(),
                           "machine": platform.machine(), "platform": platform.platform()}

    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        print(f"AlignAIR {rep['alignair_version']}")
        print(f"  python      : {rep['python']['version']} ({rep['python']['executable']})")
        t = rep["core"]["torch"]
        print(f"  torch       : {t['version']}  | CUDA available: {t['cuda']} ({t['device']})"
              if t["present"] else f"  torch       : MISSING ({t['error']})")
        g = rep["core"]["GenAIRR"]
        print(f"  GenAIRR     : {g['version']}" if g["present"] else f"  GenAIRR     : MISSING ({g['error']})")
        pp = rep["optional"]["parasail"]
        print("  parasail    : present (fast V reader available via --v-reader parasail)"
              if pp["present"] else
              "  parasail    : absent (optional; install AlignAIR[reader] for the fast V reader)")
        ac = rep["optional"]["argcomplete"]
        print("  argcomplete : present (run `alignair completion` to enable tab completion)"
              if ac["present"] else "  argcomplete : absent (optional; AlignAIR[cli] for tab completion)")
        if "model" in rep:
            m = rep["model"]
            print(f"  model       : {m['status'].upper() if m['status']=='not found' else m['status']} ({m['path']})")
        if args.verbose:
            print(f"  platform    : {rep['platform']['platform']}")
        print("status: OK" if rep["ok"] else "status: PROBLEMS FOUND")
    raise SystemExit(0 if rep["ok"] else 1)


def cmd_bundle(args) -> None:
    """Package a raw {model, config} checkpoint (+ optional calibration) into a versioned,
    fingerprinted DNAlignAIR bundle directory."""
    import torch
    from .config.dnalignair_config import DNAlignAIRConfig
    from .core.dnalignair import DNAlignAIR
    from .serialization.dnalignair_bundle import save_dnalignair_bundle
    ckpt = torch.load(args.model, map_location="cpu")
    cfg = ckpt["config"]
    cfg = DNAlignAIRConfig(**cfg) if isinstance(cfg, dict) else cfg
    model = DNAlignAIR(cfg)
    model.load_state_dict(ckpt["model"])
    calibration = json.load(open(args.calibration)) if args.calibration else None
    dataconfigs = args.dataconfig or ["HUMAN_IGH_OGRDB"]
    path = save_dnalignair_bundle(args.output, model=model, dataconfigs=dataconfigs,
                                  locus=args.locus or "IGH", calibration=calibration, notes=args.notes)
    print(f"wrote DNAlignAIR bundle -> {path}")


_TRAIN_PRESETS = {
    # name:    (d_model, n_layers, nhead, steps, batch)   — override steps with --steps
    "smoke":   (64, 2, 4, 300, 32),     # ~minutes: "does my reference train at all"
    "desktop": (128, 4, 8, 3000, 32),   # a modest, usable model
    "standard": (256, 8, 8, 8000, 32),  # paper-grade (hours on a GPU)
}

# Rough call-accuracy you can expect AFTER training each preset (human IGH; varies by reference,
# read length, and SHM). Approximate guidance so `--plan` sets expectations before GPU hours.
_PRESET_EXPECTATIONS = {
    "smoke":    "sanity only — calls are NOT accurate (tiny model meant to prove the pipeline runs)",
    "desktop":  "approx  V 0.80-0.88  J 0.90-0.95  D 0.45-0.60  (hard eval; clean is higher)",
    "standard": "approx  V 0.90-0.95  J 0.95-0.98  D 0.55-0.65  (hard eval; paper-grade)",
}


def _genairr_dataconfigs():
    """Sorted names of the built-in GenAIRR DataConfigs (usable as --reference)."""
    import GenAIRR.data as gdata
    return sorted(n for n in dir(gdata) if n.isupper() and not n.startswith("_"))


def _genairr_chain_types():
    """Valid --chain-type values for custom-FASTA references."""
    try:
        from GenAIRR.dataconfig.enums import ChainType
        return [c.name for c in ChainType]
    except Exception:
        return ["BCR_HEAVY", "BCR_LIGHT_KAPPA", "BCR_LIGHT_LAMBDA",
                "TCR_ALPHA", "TCR_BETA", "TCR_GAMMA", "TCR_DELTA"]


def _format_dataconfig_list():
    """The built-in references grouped by species, locus suffixes only (compact)."""
    by_species = {}
    for n in _genairr_dataconfigs():
        sp, _, rest = n.partition("_")
        by_species.setdefault(sp, []).append(rest or n)
    return "\n".join(f"  {sp:9s} {' '.join(by_species[sp])}" for sp in sorted(by_species))


def _resolve_train_reference(args):
    """Return (dataconfig, reference_set, bundle_ref) where bundle_ref is either
    {'dataconfigs': [name]} (built-in) or {'reference_set': rs} (custom, embedded)."""
    import GenAIRR.data as gdata
    from .reference.reference_set import ReferenceSet
    if args.v_fasta or args.j_fasta or args.d_fasta:        # custom reference from FASTA
        if not (args.v_fasta and args.j_fasta):
            raise SystemExit("error: --v-fasta and --j-fasta are required to build a custom reference")
        valid_ct = _genairr_chain_types()
        if not args.chain_type:
            raise SystemExit("error: --chain-type is required with FASTA inputs. valid: "
                             + ", ".join(valid_ct))
        if args.chain_type not in valid_ct:
            raise SystemExit(f"error: unknown --chain-type '{args.chain_type}'. valid: "
                             + ", ".join(valid_ct))
        from GenAIRR.cartridge_builder import ReferenceCartridgeBuilder
        for f in (args.v_fasta, args.j_fasta, args.d_fasta):
            if f and not os.path.exists(f):
                raise SystemExit(f"error: FASTA not found: {f}")
        dc = ReferenceCartridgeBuilder.from_fasta(
            v_fasta=args.v_fasta, j_fasta=args.j_fasta, d_fasta=args.d_fasta,
            chain_type=args.chain_type).build()
        rs = ReferenceSet.from_dataconfigs(dc)
        return dc, rs, {"reference_set": rs}
    if not args.reference:
        raise SystemExit("error: provide --reference <DATACONFIG_NAME> (see `alignair reference list`) "
                         "or build a custom one with --v-fasta/--j-fasta and --chain-type")
    if not hasattr(gdata, args.reference):
        import difflib
        names = _genairr_dataconfigs()
        near = difflib.get_close_matches(args.reference.upper(), names, n=5)
        hint = f" did you mean: {', '.join(near)}?" if near else ""
        raise SystemExit(f"error: unknown GenAIRR DataConfig '{args.reference}'.{hint}\n"
                         f"  {len(names)} built-in references available — list them with "
                         f"`alignair reference list`,\n"
                         f"  or build a custom one with --v-fasta/--j-fasta and --chain-type.")
    dc = getattr(gdata, args.reference)
    rs = ReferenceSet.from_dataconfigs(dc)
    return dc, rs, {"dataconfigs": [args.reference]}


def _fmt_dur(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


def _calibration_records(dc, n_per: int, seed: int, allow_curatable: bool):
    """A representative GenAIRR-simulated mix (clean→hard full reads, heavy-SHM tail, fragments)
    for fitting the equivalence-set calibration — mirrors a realistic deployment distribution."""
    from .gym.gym import build_experiment
    from .gym.curriculum import Curriculum
    from .gym.crop import crop_record
    cur = Curriculum()
    strata = [(0.0, None, None), (0.5, None, None), (1.0, None, None),
              (1.0, None, {"mutation_rate": 0.25}), (1.0, 120, None), (1.0, 80, None)]
    recs = []
    for j, (p, crop, ov) in enumerate(strata):
        params = cur.params(p)
        if ov:
            params.update(ov)
        exp = build_experiment(dc, params, allow_curatable=allow_curatable)
        rr = list(exp.stream_records(n=n_per, seed=seed + j))
        recs += [crop_record(r, crop) for r in rr] if crop else rr
    return recs


def cmd_train(args) -> None:
    import json as _json
    import time
    import torch
    from .config.dnalignair_config import DNAlignAIRConfig
    from .core.dnalignair import DNAlignAIR
    from .losses.dnalignair_loss import DNAlignAIRLoss
    from .gym.gym import AlignAIRGym
    from .training.gym_trainer import GymTrainer
    from .serialization.dnalignair_bundle import save_dnalignair_bundle

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    d_model, n_layers, nhead, preset_steps, preset_batch = _TRAIN_PRESETS[args.preset]
    steps = args.steps or preset_steps
    batch = args.batch or preset_batch
    out = args.out
    os.makedirs(out, exist_ok=True)
    locus = args.locus or "IGH"

    ckpt_path = os.path.join(out, "checkpoint.pt")
    resume_ckpt = None
    if args.resume:
        if not os.path.exists(ckpt_path):
            raise SystemExit(f"error: --resume but no checkpoint found at {ckpt_path}")
        resume_ckpt = torch.load(ckpt_path, map_location=device)

    dc, rs, bundle_ref = _resolve_train_reference(args)
    genes = ["v", "j"] + (["d"] if rs.has_d else [])
    nv = len(rs.gene("V").names); nj = len(rs.gene("J").names)
    nd = len(rs.gene("D").names) if rs.has_d else 0
    ref_label = (", ".join(bundle_ref["dataconfigs"]) if "dataconfigs" in bundle_ref
                 else f"custom FASTA ({locus})")

    if resume_ckpt is not None:               # rebuild from the checkpoint's own config
        model = DNAlignAIR(DNAlignAIRConfig(**resume_ckpt["config"])).to(device)
        model.load_state_dict(resume_ckpt["model"])
    elif args.base_model:
        from .api import load_model
        model = load_model(args.base_model, device).model
    else:
        model = DNAlignAIR(DNAlignAIRConfig(d_model=d_model, n_layers=n_layers, nhead=nhead)).to(device)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    gym = AlignAIRGym([dc], rs, seed=args.seed, allow_curatable=args.allow_curatable)
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=args.lr, batch_size=batch)
    start_step = 0
    if resume_ckpt is not None:                # restore optimizer + loss weights + curriculum position
        if resume_ckpt.get("loss_fn"):
            loss_fn.load_state_dict(resume_ckpt["loss_fn"])
        if resume_ckpt.get("optimizer"):
            trainer.optimizer.load_state_dict(resume_ckpt["optimizer"])
        start_step = int(resume_ckpt.get("step", 0))
        trainer._global_step = int(resume_ckpt.get("global_step", start_step))
    n_params = sum(p.numel() for p in model.parameters())

    if args.plan:                                  # preflight / dry-run: surface risks + ETA, no training
        vanc = len(rs.gene("V").anchors or {}); janc = len(rs.gene("J").anchors or {})
        anchors_ok = vanc >= nv and janc >= nj
        warnings = []
        if not anchors_ok:
            warnings.append(
                f"{nv - vanc} V and {nj - janc} J alleles lack a junction anchor. Anchors mark the "
                f"conserved Cys-104 (V) / Trp-Phe-118 (J) codons used to derive the junction/CDR3; "
                f"without them those reads train fine but get no junction, and simulation needs "
                f"--allow-curatable. Fix by adding anchors to the reference, or pass --allow-curatable.")
        if device == "cpu":
            warnings.append("no GPU detected -> training will be slow; a GPU is strongly recommended")
        if args.preset == "smoke" and not args.steps:
            warnings.append("preset 'smoke' is a pipeline sanity check only — use 'desktop' or "
                            "'standard' for a usable model")
        # quick timed estimate + peak-memory probe: warm up 1 step, then time/measure a few
        per_step = mem_note = None
        try:
            trainer.fit(total_steps=1, global_total=steps, progress=False)   # warm up (alloc caches)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            import psutil
            rss0 = psutil.Process().memory_info().rss
            t = time.time(); trainer.fit(total_steps=3, global_total=steps, progress=False)
            per_step = (time.time() - t) / 3
            if device == "cuda":
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                mem_note = f"~{peak:.2f} GB peak / {total:.1f} GB on {torch.cuda.get_device_name(0)}"
                if peak > 0.9 * total:
                    warnings.append("estimated GPU memory is close to the device limit -> lower --batch "
                                    "or pick a smaller --preset if you hit OOM")
            else:
                mem_note = f"~{psutil.Process().memory_info().rss / 1e9:.2f} GB RAM (RSS after warmup)"
        except Exception as e:
            warnings.append(f"could not time/measure a training step: {e}")

        ref_kind = ("built-in DataConfig" if "dataconfigs" in bundle_ref
                    else f"custom FASTA (chain_type={args.chain_type})")
        print("AlignAIR train -- plan (dry run, no training performed)")
        print(f"  reference : {ref_label}  [{ref_kind}]  (V={nv} D={nd} J={nj}, locus {locus})")
        print(f"  anchors   : V {vanc}/{nv}, J {janc}/{nj}"
              + ("  (junctions OK)" if anchors_ok else "  (incomplete — see WARNING)"))
        print(f"  model     : {n_params/1e6:.2f}M params (d_model={model.config.d_model}, "
              f"layers={model.config.n_layers}, preset={args.preset})")
        if per_step is not None:
            eta = per_step * steps
            print(f"  time est. : {steps} steps x {per_step:.2f}s/step ~= {_fmt_dur(eta)} on {device}"
                  f"  (+ calibration unless --no-calibrate)")
        else:
            print(f"  time est. : {steps} steps on {device} (timing unavailable)")
        if mem_note:
            print(f"  memory est: {mem_note}  (batch={batch})")
        print(f"  expected  : {_PRESET_EXPECTATIONS.get(args.preset, 'n/a')}")
        print(f"  outputs   : {out}/bundle, {out}/model_card.md, {out}/validation_report.json, "
              f"{out}/checkpoint.pt")
        for w in warnings:
            print(f"  WARNING   : {w}")
        print("\nrun the same command without --plan to train.")
        return

    print(f"AlignAIR train")
    print(f"  reference : {ref_label}  (V={nv} D={nd} J={nj}, locus {locus})")
    print(f"  model     : {n_params/1e6:.2f}M params (d_model={model.config.d_model}, "
          f"layers={model.config.n_layers})"
          + (f"  resumed from step {start_step}" if resume_ckpt is not None
             else (f"  fine-tuned from {args.base_model}" if args.base_model else "")))
    print(f"  run       : preset={args.preset} steps={steps} batch={batch} seed={args.seed} device={device}")
    print(f"  output    : {out}/\n")
    print("training:" if start_step < steps else "training: (already at target steps, finalizing)")

    t0 = time.time()
    step = start_step
    while step < steps:
        chunk = min(args.eval_every, steps - step)
        trainer.fit(total_steps=chunk, global_total=steps, progress=False)
        step += chunk
        ev = trainer.evaluate(n_batches=3)
        elapsed = time.time() - t0
        done = step - start_step                       # steps completed THIS run (for ETA)
        eta = elapsed / done * (steps - step) if done else 0
        call = " ".join(f"{g.upper()}={ev[f'{g}_call']:.2f}" for g in genes)
        print(f"  {step:>6}/{steps} ({step/steps*100:3.0f}%)  {_fmt_dur(elapsed):>7}  "
              f"eta {_fmt_dur(eta):>6}  loss={ev['loss']:6.3f}  region={ev['region_acc']:.3f}  {call}",
              flush=True)
        torch.save({"model": model.state_dict(), "config": model.config.to_dict(), "step": step,
                    "global_step": trainer._global_step, "optimizer": trainer.optimizer.state_dict(),
                    "loss_fn": loss_fn.state_dict()}, ckpt_path)   # full state for --resume

    # ---- equivalence-set calibration (folded in so trained models ship good sets) ----
    calibration = None
    if not args.no_calibrate:
        print("\ncalibration (equivalence sets, F1-objective on a representative mix):")
        from .benchmark.evaluation.allele_calibration import (
            collect_calibration_rows, fit_calibration, fit_contaminant_tau)
        cal_recs = _calibration_records(dc, args.calib_n, args.seed + 1000, args.allow_curatable)
        rows, gate = collect_calibration_rows(model, rs, cal_recs, topk=32, device=device,
                                              genes=tuple(genes))
        calibration = fit_calibration(rows, objective="f1")
        tau = fit_contaminant_tau(gate)
        if tau is not None:
            calibration["contaminant"] = {"tau": tau, "fpr_target": 0.02, "n": len(gate)}
        for G, c in calibration.items():
            if G == "contaminant":
                print(f"  contaminant gate : tau={c['tau']:.3f}")
            else:
                print(f"  {G}: set_size={c['mean_set_size']:.2f}  recall={c['set_recall']:.3f}  "
                      f"f1={c.get('set_f1', float('nan')):.3f}  (T={c['temperature']:.2f} eps={c['epsilon']:.2f})")

    # ---- final validation ----
    ev = trainer.evaluate(n_batches=8)
    ev_clean = trainer.evaluate(n_batches=8, p=0.0)
    wall = time.time() - t0
    report = {"reference": bundle_ref.get("dataconfigs", ["custom-fasta"]), "locus": locus,
              "steps": steps, "preset": args.preset, "seed": args.seed, "n_params": n_params,
              "wall_seconds": round(wall, 1), "v_count": nv, "j_count": nj, "d_count": nd,
              "eval_hard": {g: round(ev[f"{g}_call"], 4) for g in genes},
              "eval_clean": {g: round(ev_clean[f"{g}_call"], 4) for g in genes},
              "region_acc": round(ev["region_acc"], 4), "orientation_acc": round(ev["orient_acc"], 4),
              "calibration": calibration}
    _json.dump(report, open(os.path.join(out, "validation_report.json"), "w"), indent=2)
    print("\nvalidation (final, GenAIRR-simulated):")
    print("  call accuracy   " + "  ".join(f"{g.upper()}: {report['eval_hard'][g]:.3f} (hard) "
                                           f"{report['eval_clean'][g]:.3f} (clean)" for g in genes))
    print(f"  region_acc={report['region_acc']:.3f}  orientation_acc={report['orientation_acc']:.3f}")

    # ---- bundle + model card ----
    bundle_dir = os.path.join(out, "bundle")
    save_dnalignair_bundle(bundle_dir, model=model, locus=locus,
                           dataconfigs=bundle_ref.get("dataconfigs"),
                           reference_set=bundle_ref.get("reference_set"), calibration=calibration,
                           notes=f"alignair train preset={args.preset} steps={steps} seed={args.seed}",
                           training_meta={"preset": args.preset, "steps": steps, "seed": args.seed,
                                          "lr": args.lr, "batch": getattr(args, "batch", None),
                                          "base_model": args.base_model, "device": device,
                                          "n_params": n_params, "wall_seconds": round(wall, 1),
                                          "reference": ref_label,
                                          "eval_hard": report["eval_hard"]})
    card = (f"# AlignAIR model\n\n"
            f"- **Reference**: {ref_label}\n- **Locus**: {locus}\n"
            f"- **Preset**: {args.preset} (steps={steps}, params={n_params/1e6:.2f}M, seed={args.seed})\n"
            f"- **Calibration**: {'fitted (F1)' if calibration else 'none'}\n"
            f"- **Trained**: {_fmt_dur(wall)} on {device}\n\n"
            f"## Validation (GenAIRR-simulated)\n\n"
            f"| gene | call acc (hard) | call acc (clean) |\n|---|---|---|\n"
            + "".join(f"| {g.upper()} | {report['eval_hard'][g]:.3f} | {report['eval_clean'][g]:.3f} |\n"
                      for g in genes)
            + f"\nregion_acc={report['region_acc']:.3f}  orientation_acc={report['orientation_acc']:.3f}\n\n"
            f"## Use\n\n```bash\nalignair predict reads.fasta -o out.tsv --model {bundle_dir}\n```\n")
    open(os.path.join(out, "model_card.md"), "w").write(card)

    print(f"\ndone in {_fmt_dur(wall)}.")
    print(f"  bundle            -> {bundle_dir}")
    print(f"  model card        -> {os.path.join(out, 'model_card.md')}")
    print(f"  validation report -> {os.path.join(out, 'validation_report.json')}")
    print(f"  try it: alignair predict reads.fasta -o out.tsv --model {bundle_dir}")


def cmd_demo(args) -> None:
    """One offline command that proves AlignAIR works end-to-end with NO published model and NO
    network: trains a tiny demo model, predicts on simulated reads, validates the AIRR output, and
    runs the dynamic-genotype path. The model is intentionally tiny (not production quality)."""
    import GenAIRR.data as gdata
    from .reference.reference_set import ReferenceSet
    from .gym.gym import build_experiment
    from .gym.curriculum import Curriculum

    out = args.out
    os.makedirs(out, exist_ok=True)
    parser = build_parser()
    print("AlignAIR demo — proving the full pipeline offline (no published model, no network).")
    print(f"  output: {out}/\n")

    # 1) tiny model
    print("[1/4] training a TINY demo model (not production quality) ...")
    cmd_train(parser.parse_args(
        ["train", "--reference", "HUMAN_IGH_OGRDB", "--out", out, "--preset", "smoke",
         "--steps", str(args.steps), "--eval-every", str(args.steps), "--no-calibrate"]
        + (["--device", args.device] if args.device else [])))
    bundle = os.path.join(out, "bundle")

    # 2) simulate a few example reads + a donor genotype (offline, from GenAIRR)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    exp = build_experiment(gdata.HUMAN_IGH_OGRDB, Curriculum().params(0.3))
    recs = list(exp.stream_records(n=10, seed=1))
    reads_path = os.path.join(out, "demo_reads.fasta")
    with open(reads_path, "w") as f:
        for i, r in enumerate(recs):
            f.write(f">read{i+1}\n{r['sequence']}\n")
    donor = os.path.join(out, "donor_genotype.yaml")
    truth_v = sorted({str(r.get("v_call", "")).split(",")[0] for r in recs if r.get("v_call")})
    rs.subset({"v": truth_v + rs.gene("V").names[:8], "d": rs.gene("D").names[:10],
               "j": rs.gene("J").names}).to_yaml(donor)

    # 3) predict + validate
    print("\n[2/4] aligning example reads ...")
    out_tsv = os.path.join(out, "demo.tsv")
    dev = (["--device", args.device] if args.device else [])
    cmd_predict(parser.parse_args(["predict", reads_path, "-o", out_tsv, "--model", bundle] + dev))
    print("\n[3/4] validating AIRR output ...")
    try:
        cmd_validate_airr(parser.parse_args(["validate-airr", out_tsv]))
    except SystemExit:
        pass

    # 4) dynamic genotype
    print("\n[4/4] dynamic genotype — aligning against a donor reference (subset + the truth alleles) ...")
    cmd_predict(parser.parse_args(
        ["predict", reads_path, "-o", os.path.join(out, "demo_donor.tsv"),
         "--model", bundle, "--genotype", donor] + dev))

    print("\nAlignAIR works end-to-end. NOTE: this was a TINY demo model — calls are NOT accurate.")
    print("Next: train a real model (`alignair train --reference HUMAN_IGH_OGRDB -o my_model "
          "--preset desktop`) or use a published bundle when available.")
    print(f"Artifacts in {out}/: bundle/, demo.tsv, demo_donor.tsv, demo_reads.fasta, donor_genotype.yaml")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="alignair",
                                 description="AlignAIR — neural IG/TCR sequence aligner")
    ap.add_argument("--version", action="version", version=f"AlignAIR {_version()}")
    sub = ap.add_subparsers(required=True, dest="command")
    pr = sub.add_parser("predict", help="align reads and write an AIRR rearrangement TSV")
    pr.add_argument("input", help="FASTA/FASTQ/CSV/TSV/TXT (optionally .gz) of reads, or '-' for stdin")
    pr.add_argument("-o", "--output", required=True,
                    help="output AIRR rearrangement TSV, or '-' for stdout")
    pr.add_argument("--sequence-column", default=None, help="CSV/TSV: column holding the read sequence")
    pr.add_argument("--id-column", default=None, help="CSV/TSV: column holding the sequence id")
    pr.add_argument("--metadata", default=None,
                    help="per-read metadata CSV/TSV to preserve into output (e.g. 10x "
                         "filtered_contig_annotations.csv, or an AIRR TSV); joined by read id")
    pr.add_argument("--metadata-id", default=None,
                    help="id column in --metadata (default: auto sequence_id/contig_id/cell_id)")
    pr.add_argument("--keep-columns", default=None,
                    help="comma-separated metadata columns to carry (default: known 10x/AIRR set)")
    pr.add_argument("--seed", type=int, default=0, help="random seed (recorded in run.json)")
    pr.add_argument("--model", required=True,
                    help="a bundle dir / raw .pt checkpoint, a catalog id (alignair model list), "
                         "or an org/name Hugging Face repo id (auto-downloaded)")
    pr.add_argument("--genotype", default=None,
                    help="genotype file (YAML or FASTA) used as the reference "
                         "(allele subset and/or novel alleles)")
    pr.add_argument("--calibration", default=None,
                    help="allele-set calibration JSON (overrides a bundled calibration)")
    pr.add_argument("--dataconfig", default="HUMAN_IGH_OGRDB",
                    help="GenAIRR DataConfig name for the reference (raw checkpoint, no --genotype)")
    pr.add_argument("--locus", default=None, help="locus label (default: bundle's, else IGH)")
    pr.add_argument("--batch", type=int, default=64)
    pr.add_argument("--chunk-size", type=int, default=20000,
                    help="reads held in memory per streaming chunk (repertoire-scale; default 20000)")
    pr.add_argument("--device", default=None, help="cuda|cpu (auto if unset)")
    pr.add_argument("--v-reader", default="learned", choices=["learned", "parasail"],
                    help="V allele reader: learned (default) or parasail (faster+sharper; needs AlignAIR[reader])")
    pr.add_argument("--quiet", action="store_true", help="suppress progress output")
    pr.add_argument("--no-provenance", action="store_true",
                    help="do not write the <output>.run.json provenance sidecar")
    pr.add_argument("--no-full-alignment", action="store_true",
                    help="skip the parasail gapped alignment (exact cigars / germline_alignment / "
                         "identity); use the faster coordinate-derived approximation")
    pr.add_argument("--force-locus-mismatch", action="store_true",
                    help="proceed even if the model's locus does not match the reference (default: error)")
    pr.set_defaults(func=cmd_predict)

    ba = sub.add_parser("batch", help="align many samples from a manifest with one model load")
    ba.add_argument("--manifest", required=True,
                    help="CSV/TSV with columns sample_id,input (optional per-row: genotype, metadata); "
                         "relative paths resolve against the manifest's directory")
    ba.add_argument("-o", "--output", required=True,
                    help="output directory: one <sample_id>.tsv per sample + manifest_summary.tsv/json")
    ba.add_argument("--model", required=True,
                    help="bundle dir / .pt checkpoint / catalog id / HF repo id (loaded once for all samples)")
    ba.add_argument("--genotype", default=None,
                    help="default genotype (YAML/FASTA) for rows that don't set their own")
    ba.add_argument("--metadata-id", default=None, help="id column in each row's --metadata file")
    ba.add_argument("--keep-columns", default=None,
                    help="comma-separated metadata columns to carry (default: known 10x/AIRR set)")
    ba.add_argument("--sequence-column", default=None, help="CSV/TSV input: column holding the sequence")
    ba.add_argument("--id-column", default=None, help="CSV/TSV input: column holding the sequence id")
    ba.add_argument("--calibration", default=None, help="allele-set calibration JSON (overrides bundle)")
    ba.add_argument("--dataconfig", default="HUMAN_IGH_OGRDB",
                    help="GenAIRR DataConfig for the reference (raw checkpoint, no genotype)")
    ba.add_argument("--locus", default=None, help="locus label (default: bundle's, else IGH)")
    ba.add_argument("--batch", type=int, default=64)
    ba.add_argument("--chunk-size", type=int, default=20000,
                    help="reads held in memory per streaming chunk (default 20000)")
    ba.add_argument("--device", default=None, help="cuda|cpu (auto if unset)")
    ba.add_argument("--v-reader", default="learned", choices=["learned", "parasail"],
                    help="V allele reader: learned (default) or parasail")
    ba.add_argument("--seed", type=int, default=0, help="random seed (recorded in each run.json)")
    ba.add_argument("--quiet", action="store_true", help="suppress progress output")
    ba.add_argument("--no-provenance", action="store_true",
                    help="do not write per-sample <output>.run.json provenance sidecars")
    ba.add_argument("--no-full-alignment", action="store_true",
                    help="skip the parasail gapped alignment (faster coordinate-derived cigars)")
    ba.add_argument("--force-locus-mismatch", action="store_true",
                    help="proceed even if the model's locus does not match the reference")
    ba.set_defaults(func=cmd_batch)

    va = sub.add_parser("validate-airr", help="validate a rearrangement TSV against the AIRR-C schema")
    va.add_argument("file", help="AIRR rearrangement TSV to validate")
    va.set_defaults(func=cmd_validate_airr)

    cp = sub.add_parser("compare", help="agreement report between two AIRR TSVs (e.g. AlignAIR vs IgBLAST)")
    cp.add_argument("--a", required=True, help="AIRR TSV for tool A (AlignAIR, for set-rescue)")
    cp.add_argument("--b", required=True, help="AIRR TSV for tool B (e.g. IgBLAST/MiXCR exportAirr)")
    cp.add_argument("--a-name", default="AlignAIR")
    cp.add_argument("--b-name", default="other")
    cp.add_argument("--out", default=None, help="write the Markdown report here (default: stdout)")
    cp.add_argument("--json", default=None, help="also write the report as JSON here")
    cp.set_defaults(func=cmd_compare)

    dr = sub.add_parser("doctor", help="check the environment (Python, torch+CUDA, GenAIRR, parasail)")
    dr.add_argument("--model", default=None, help="optionally verify a model bundle/checkpoint resolves")
    dr.add_argument("--json", action="store_true", help="emit a machine-readable JSON report")
    dr.add_argument("-v", "--verbose", action="store_true", help="include platform/build detail")
    dr.set_defaults(func=cmd_doctor)

    dm = sub.add_parser("demo", help="offline end-to-end trial (tiny train -> predict -> validate -> genotype)")
    dm.add_argument("-o", "--out", default="alignair_demo", help="demo output directory")
    dm.add_argument("--steps", type=int, default=60, help="tiny-model training steps (default 60)")
    dm.add_argument("--device", default=None, help="cuda|cpu (auto if unset)")
    dm.set_defaults(func=cmd_demo)

    rf = sub.add_parser("reference", help="list / validate / convert germline references")
    rsub = rf.add_subparsers(required=True, dest="reference_command")
    rls = rsub.add_parser("list", help="list built-in GenAIRR references and valid chain types")
    rls.add_argument("--json", action="store_true", help="emit names as JSON")
    rv = rsub.add_parser("validate", help="check a reference file (counts, anchors, duplicates)")
    rv.add_argument("file", help="genotype YAML or germline FASTA")
    rc = rsub.add_parser("convert", help="convert between genotype YAML and FASTA")
    rc.add_argument("file", help="input reference (YAML or FASTA)")
    rc.add_argument("-o", "--out", required=True, help="output path (.yaml or .fasta sets the format)")
    rt = rsub.add_parser("template", help="dump a built-in reference to YAML as a genotype starting point")
    rt.add_argument("reference", help="GenAIRR DataConfig name (e.g. HUMAN_IGH_OGRDB)")
    rt.add_argument("-o", "--out", required=True, help="output genotype YAML to edit")
    rf.set_defaults(func=cmd_reference)

    md = sub.add_parser("model", help="list / download / inspect pretrained models")
    msub = md.add_subparsers(required=True, dest="model_command")
    ml = msub.add_parser("list", help="list the pretrained model catalog")
    ml.add_argument("--json", action="store_true", help="emit the catalog as JSON")
    mdl = msub.add_parser("download", help="download a model bundle (catalog id or HF repo id)")
    mdl.add_argument("id", help="catalog id (alignair model list) or org/name Hugging Face repo id")
    mdl.add_argument("--dest", default=None, help="download into this directory (default: HF cache)")
    mi = msub.add_parser("inspect", help="show a model bundle's reference, config, and metadata")
    mi.add_argument("id", help="a bundle path, catalog id, or org/name Hugging Face repo id")
    mi.add_argument("--json", action="store_true", help="emit bundle metadata as JSON")
    md.set_defaults(func=cmd_model)

    tr = sub.add_parser("train", help="train an AlignAIR model for your own reference / species")
    tr.add_argument("--reference", default=None,
                    help="a built-in GenAIRR DataConfig name (e.g. HUMAN_IGH_OGRDB, MOUSE_IGH_IMGT)")
    tr.add_argument("--v-fasta", default=None, help="custom reference: V germline FASTA")
    tr.add_argument("--j-fasta", default=None, help="custom reference: J germline FASTA")
    tr.add_argument("--d-fasta", default=None, help="custom reference: D germline FASTA (heavy/beta/delta)")
    tr.add_argument("--chain-type", default=None,
                    help="custom reference: GenAIRR chain type (e.g. BCR_HEAVY, BCR_LIGHT_KAPPA, TCR_BETA)")
    tr.add_argument("-o", "--out", required=True, help="output run directory (bundle + reports)")
    tr.add_argument("--preset", choices=list(_TRAIN_PRESETS), default="desktop",
                    help="size/length preset (smoke|desktop|standard); override length with --steps")
    tr.add_argument("--steps", type=int, default=None, help="training steps (overrides preset)")
    tr.add_argument("--batch", type=int, default=None, help="batch size (overrides preset)")
    tr.add_argument("--base-model", default=None,
                    help="fine-tune from this bundle/checkpoint instead of training from scratch")
    tr.add_argument("--resume", action="store_true",
                    help="resume from <out>/checkpoint.pt (restores weights, optimizer, and "
                         "curriculum position); continue to --steps")
    tr.add_argument("--plan", action="store_true",
                    help="dry run: show reference checks, anchor coverage, model size, a timed "
                         "wall-time estimate, outputs, and warnings -- without training")
    tr.add_argument("--locus", default="IGH")
    tr.add_argument("--allow-curatable", action="store_true",
                    help="permit simulation from references with curatable issues (e.g. custom "
                         "FASTA alleles with no detected anchor); required for some custom references")
    tr.add_argument("--no-calibrate", action="store_true",
                    help="skip fitting the equivalence-set calibration after training")
    tr.add_argument("--calib-n", type=int, default=200,
                    help="records per stratum for calibration (default 200)")
    tr.add_argument("--lr", type=float, default=5e-4)
    tr.add_argument("--eval-every", type=int, default=200)
    tr.add_argument("--device", default=None, help="cuda|cpu (auto if unset)")
    tr.add_argument("--seed", type=int, default=0)
    tr.set_defaults(func=cmd_train)

    bd = sub.add_parser("bundle", help="package a raw checkpoint into a versioned bundle")
    bd.add_argument("--model", required=True, help="raw .pt checkpoint {model, config}")
    bd.add_argument("-o", "--output", required=True, help="bundle directory to create")
    bd.add_argument("--calibration", default=None, help="allele-set calibration JSON to include")
    bd.add_argument("--dataconfig", action="append", default=None,
                    help="GenAIRR DataConfig name(s) for the default reference (repeatable)")
    bd.add_argument("--locus", default=None)
    bd.add_argument("--notes", default=None)
    bd.set_defaults(func=cmd_bundle)

    co = sub.add_parser("completion", help="print shell tab-completion setup instructions")
    co.add_argument("shell", nargs="?", default="bash", choices=["bash", "zsh", "fish"],
                    help="target shell (default: bash)")
    co.set_defaults(func=cmd_completion)
    return ap


def cmd_completion(args) -> None:
    """Print the one line to enable tab completion for the chosen shell (needs argcomplete,
    included in AlignAIR[cli])."""
    try:
        import argcomplete  # noqa: F401
    except ImportError:
        raise SystemExit("error: shell completion needs `argcomplete` — install with "
                         "`pip install \"AlignAIR[cli]\"` or `pip install argcomplete`")
    if args.shell == "bash":
        print('eval "$(register-python-argcomplete alignair)"   # add to ~/.bashrc')
    elif args.shell == "zsh":
        print("autoload -U bashcompinit && bashcompinit            # add to ~/.zshrc")
        print('eval "$(register-python-argcomplete alignair)"     # add to ~/.zshrc')
    else:  # fish
        print("register-python-argcomplete --shell fish alignair > "
              "~/.config/fish/completions/alignair.fish")


def main(argv=None) -> None:
    parser = build_parser()
    try:                                  # optional: tab completion when argcomplete is installed
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
