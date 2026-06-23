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


def _load_model(model_path, device):
    """Load a DNAlignAIR from either a versioned bundle directory or a raw {model, config}
    .pt checkpoint. Returns (model, dataconfigs, locus, calibration, reference_set)."""
    import torch
    from .config.dnalignair_config import DNAlignAIRConfig
    from .core.dnalignair import DNAlignAIR
    from .serialization.dnalignair_bundle import is_bundle, load_dnalignair_bundle
    if is_bundle(model_path):
        b = load_dnalignair_bundle(model_path, build=True, device=device)
        return b["model"], b["dataconfigs"], b["locus"], b["calibration"], b.get("reference_set")
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt["config"]
    cfg = DNAlignAIRConfig(**cfg) if isinstance(cfg, dict) else cfg
    model = DNAlignAIR(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, None, None, None, None


def cmd_predict(args) -> None:
    import torch
    import GenAIRR.data as gdata
    from .reference.reference_set import ReferenceSet
    from .inference.dnalignair_infer import predict_reads, canonicalize_sequence

    def log(msg):
        if not args.quiet:
            print(msg, file=sys.stderr, flush=True)      # progress -> stderr (keeps stdout clean)

    torch.manual_seed(args.seed)
    from .hub import resolve_model
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = resolve_model(args.model)             # local path, catalog id, or HF repo id
    model, b_dataconfigs, b_locus, b_calibration, b_reference = _load_model(model_path, device)
    locus = args.locus or b_locus or "IGH"
    log(f"device: {device}  |  model: {args.model}")

    if args.genotype:
        if not os.path.exists(args.genotype):
            raise SystemExit(f"error: genotype file not found: {args.genotype}")
        ext = os.path.splitext(args.genotype)[1].lower()
        loader = ReferenceSet.from_fasta if ext in (".fasta", ".fa", ".fna", ".faa") else ReferenceSet.from_yaml
        rs = loader(args.genotype)
        log(f"reference: genotype {args.genotype} "
            f"(V={len(rs.gene('V').names)}"
            f"{', D=' + str(len(rs.gene('D').names)) if rs.has_d else ''}, J={len(rs.gene('J').names)})")
    elif b_reference is not None:                     # bundle embeds its own (custom) reference
        rs = b_reference
        ref_desc = f"bundled ({len(rs.gene('V').names)} V alleles)"
        log(f"reference: bundled (V={len(rs.gene('V').names)}"
            f"{', D=' + str(len(rs.gene('D').names)) if rs.has_d else ''}, J={len(rs.gene('J').names)})")
    else:
        names = b_dataconfigs or [args.dataconfig]
        try:
            rs = ReferenceSet.from_dataconfigs(*[getattr(gdata, n) for n in names])
        except AttributeError as e:
            raise SystemExit(f"error: unknown GenAIRR DataConfig in {names}: {e}")
        ref_desc = ", ".join(names)
        log(f"reference: {', '.join(names)} (V={len(rs.gene('V').names)})")
    if args.genotype:
        ref_desc = f"genotype:{os.path.basename(args.genotype)}"
    # locus/chain sanity: a model declares its trained locus (in its bundle); warn if the
    # reference looks like a different locus (a heavy model + light reference produces
    # plausible-but-meaningless calls without this check).
    inferred_locus = rs.infer_locus()
    if inferred_locus and b_locus and inferred_locus != b_locus:
        log(f"WARNING: this model is for locus {b_locus}, but the reference looks like "
            f"{inferred_locus} — V/D/J calls may be meaningless; use a model and reference "
            f"for the same locus.")
    if args.locus is None and inferred_locus:
        locus = inferred_locus                  # reflect the reference's locus in the output

    calibration = b_calibration
    if args.calibration and os.path.exists(args.calibration):
        calibration = json.load(open(args.calibration))           # explicit flag overrides bundle

    # STREAMING: read -> predict -> write in bounded-memory chunks (repertoire-scale safe).
    from .io.sequence_reader import iter_sequences
    from .io.airr import AirrWriter
    to_stdout = args.output == "-"
    write_provenance = not (args.no_provenance or to_stdout)
    chunk = max(args.batch, args.chunk_size)
    bar = None if (args.quiet or to_stdout) else __import__("tqdm").tqdm(
        desc="aligning", unit="read", file=sys.stderr)
    total = n_prod = n_contam = n_dropped = 0
    try:
        if not to_stdout:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        writer = AirrWriter(args.output, locus)
    except (PermissionError, OSError) as e:
        raise SystemExit(
            f"error: cannot write output to {args.output}: {e}\n"
            f"hint: if running in Docker, mount a writable output directory and add "
            f"`--user $(id -u):$(id -g)` so files are written as you.")
    try:
        try:
            chunks = iter_sequences(args.input, chunk_size=chunk,
                                    seq_column=args.sequence_column, id_column=args.id_column)
            for ids, seqs, drp in chunks:
                n_dropped += drp
                if not seqs:
                    continue
                preds = predict_reads(model, rs, seqs, device=device, batch_size=args.batch,
                                      rerank="learned", v_reader=args.v_reader,
                                      calibration=calibration, full_alignment=not args.no_full_alignment)
                # emit the CANONICAL (forward) sequence so coords match even for reoriented reads
                canon = [canonicalize_sequence(s, p["orientation_id"]) for s, p in zip(seqs, preds)]
                writer.write(ids, canon, preds)
                total += len(preds)
                n_prod += sum(1 for p in preds if p.get("productive"))
                n_contam += sum(1 for p in preds if p.get("is_contaminant"))
                if bar is not None:
                    bar.update(len(preds))
        except (ValueError, FileNotFoundError) as e:
            raise SystemExit(f"error: {e}")
    finally:
        writer.close()
        if bar is not None:
            bar.close()
    if total == 0:
        raise SystemExit("error: no valid sequences to align")
    if write_provenance:
        from .io.sequence_reader import _detect_format
        info = {"n_read": total + n_dropped, "n_dropped": n_dropped,
                "format": _detect_format(args.input)}
        _write_provenance(args.output + ".run.json", args=args, model_path=model_path,
                          device=device, info=info, ref_desc=ref_desc)
    if not to_stdout:
        log(f"wrote {total} rearrangements ({n_dropped} dropped) -> {args.output}"
            + (f"  (+ {os.path.basename(args.output)}.run.json)" if write_provenance else ""))
        log(f"summary: {total} aligned | {n_prod} productive ({n_prod/max(total,1)*100:.0f}%)"
            f"{f' | {n_contam} flagged out-of-scope' if n_contam else ''} | locus {locus}")


def _pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _write_provenance(path, *, args, model_path, device, info, ref_desc):
    """Write a run.json sidecar next to the output: what produced this file (AIRR Software WG
    expects run parameters to travel with the output)."""
    import datetime
    fingerprint = None
    fp = os.path.join(model_path, "fingerprint.txt") if os.path.isdir(model_path) else None
    if fp and os.path.exists(fp):
        fingerprint = open(fp).read().strip()
    prov = {
        "tool": "AlignAIR", "alignair_version": _version(),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": "alignair " + " ".join(sys.argv[1:]),
        "model": model_path, "model_fingerprint": fingerprint, "reference": ref_desc,
        "device": device, "v_reader": args.v_reader, "batch": args.batch, "seed": args.seed,
        "n_input": info.get("n_read"), "n_dropped": info.get("n_dropped"),
        "input_format": info.get("format"), "output": args.output,
        "versions": {k: _pkg_version(k) for k in ("torch", "GenAIRR", "airr", "parasail")},
    }
    with open(path, "w") as f:
        json.dump(prov, f, indent=2)


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
        if not MODEL_CATALOG:
            print("(no models in the catalog yet)")
            return
        print(f"{'id':22s} {'status':12s} {'species':8s} {'locus':6s} description")
        for mid, e in MODEL_CATALOG.items():
            status = "available" if e.get("available", True) else "not published"
            print(f"{mid:22s} {status:12s} {e.get('species',''):8s} {e.get('locus',''):6s} {e.get('description','')}")
        if any(not e.get("available", True) for e in MODEL_CATALOG.values()):
            print("\nnot-published models aren't downloadable yet — train your own (`alignair train`) "
                  "or try `alignair demo`.")
        print("\ndownload: alignair model download <id>   |   use directly: alignair predict ... --model <id>")
    elif args.model_command == "download":
        path = resolve_model(args.id, dest=args.dest)
        print(f"downloaded -> {path}")
    elif args.model_command == "inspect":
        path = resolve_model(args.id)
        if not is_bundle(path):
            raise SystemExit(f"error: {path} is not an AlignAIR bundle (raw checkpoints carry no metadata)")
        b = load_dnalignair_bundle(path, build=False)
        cfg = b["config"].to_dict()
        ref = (f"{len(b['reference_set'].gene('V'))} V (embedded)" if b.get("reference_set")
               else f"dataconfigs={b['dataconfigs']}")
        print(f"bundle: {path}")
        print(f"  locus       : {b['locus']}")
        print(f"  reference   : {ref}")
        print(f"  d_model={cfg.get('d_model')} n_layers={cfg.get('n_layers')} nhead={cfg.get('nhead')}")
        print(f"  calibration : {'yes' if b['calibration'] else 'no'}")
        print(f"  notes       : {(b['meta'] or {}).get('notes')}")


def _load_reference_file(path):
    from .reference.reference_set import ReferenceSet
    if not os.path.exists(path):
        raise SystemExit(f"error: file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    return (ReferenceSet.from_fasta(path) if ext in (".fasta", ".fa", ".fna", ".faa")
            else ReferenceSet.from_yaml(path))


def cmd_reference(args) -> None:
    """Validate or convert a germline reference (YAML genotype <-> FASTA)."""
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
    """Environment / install check: Python, PyTorch + CUDA, GenAIRR, optional parasail, and
    (optionally) whether a --model path resolves. Exit non-zero if a CORE dependency is missing."""
    ok = True
    print(f"AlignAIR {_version()}")
    print(f"  python      : {sys.version.split()[0]} ({sys.executable})")
    try:
        import torch
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "cpu only"
        print(f"  torch       : {torch.__version__}  | CUDA available: {cuda} ({dev})")
    except Exception as e:
        ok = False; print(f"  torch       : MISSING ({e})")
    try:
        import GenAIRR
        print(f"  GenAIRR     : {getattr(GenAIRR, '__version__', '?')}")
    except Exception as e:
        ok = False; print(f"  GenAIRR     : MISSING ({e})")
    try:
        import parasail  # noqa: F401
        print("  parasail    : present (fast V reader available via --v-reader parasail)")
    except Exception:
        print("  parasail    : absent (optional; install AlignAIR[reader] for the fast V reader)")
    if args.model:
        from .serialization.dnalignair_bundle import is_bundle
        if not os.path.exists(args.model):
            ok = False; print(f"  model       : NOT FOUND ({args.model})")
        else:
            print(f"  model       : {'bundle' if is_bundle(args.model) else 'raw checkpoint'} at {args.model}")
    print("status: OK" if ok else "status: PROBLEMS FOUND")
    raise SystemExit(0 if ok else 1)


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


def _resolve_train_reference(args):
    """Return (dataconfig, reference_set, bundle_ref) where bundle_ref is either
    {'dataconfigs': [name]} (built-in) or {'reference_set': rs} (custom, embedded)."""
    import GenAIRR.data as gdata
    from .reference.reference_set import ReferenceSet
    if args.v_fasta or args.j_fasta or args.d_fasta:        # custom reference from FASTA
        if not (args.v_fasta and args.j_fasta):
            raise SystemExit("error: --v-fasta and --j-fasta are required to build a custom reference")
        if not args.chain_type:
            raise SystemExit("error: --chain-type is required with FASTA inputs "
                             "(e.g. BCR_HEAVY, BCR_LIGHT_KAPPA, TCR_BETA)")
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
        raise SystemExit("error: provide --reference <DATACONFIG_NAME> or --v-fasta/--j-fasta")
    if not hasattr(gdata, args.reference):
        raise SystemExit(f"error: unknown GenAIRR DataConfig '{args.reference}'. "
                         f"List names with: python -c \"import GenAIRR.data as g; print([n for n in dir(g) if n.isupper()])\"")
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
        model, _, _, _, _ = _load_model(args.base_model, device)
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
                           notes=f"alignair train preset={args.preset} steps={steps} seed={args.seed}")
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
    pr.set_defaults(func=cmd_predict)

    va = sub.add_parser("validate-airr", help="validate a rearrangement TSV against the AIRR-C schema")
    va.add_argument("file", help="AIRR rearrangement TSV to validate")
    va.set_defaults(func=cmd_validate_airr)

    dr = sub.add_parser("doctor", help="check the environment (Python, torch+CUDA, GenAIRR, parasail)")
    dr.add_argument("--model", default=None, help="optionally verify a model bundle/checkpoint resolves")
    dr.set_defaults(func=cmd_doctor)

    dm = sub.add_parser("demo", help="offline end-to-end trial (tiny train -> predict -> validate -> genotype)")
    dm.add_argument("-o", "--out", default="alignair_demo", help="demo output directory")
    dm.add_argument("--steps", type=int, default=60, help="tiny-model training steps (default 60)")
    dm.add_argument("--device", default=None, help="cuda|cpu (auto if unset)")
    dm.set_defaults(func=cmd_demo)

    rf = sub.add_parser("reference", help="validate or convert a germline reference (YAML <-> FASTA)")
    rsub = rf.add_subparsers(required=True, dest="reference_command")
    rv = rsub.add_parser("validate", help="check a reference file (counts, anchors, duplicates)")
    rv.add_argument("file", help="genotype YAML or germline FASTA")
    rc = rsub.add_parser("convert", help="convert between genotype YAML and FASTA")
    rc.add_argument("file", help="input reference (YAML or FASTA)")
    rc.add_argument("-o", "--out", required=True, help="output path (.yaml or .fasta sets the format)")
    rf.set_defaults(func=cmd_reference)

    md = sub.add_parser("model", help="list / download / inspect pretrained models")
    msub = md.add_subparsers(required=True, dest="model_command")
    msub.add_parser("list", help="list the pretrained model catalog")
    mdl = msub.add_parser("download", help="download a model bundle (catalog id or HF repo id)")
    mdl.add_argument("id", help="catalog id (alignair model list) or org/name Hugging Face repo id")
    mdl.add_argument("--dest", default=None, help="download into this directory (default: HF cache)")
    mi = msub.add_parser("inspect", help="show a model bundle's reference, config, and metadata")
    mi.add_argument("id", help="a bundle path, catalog id, or org/name Hugging Face repo id")
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
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
