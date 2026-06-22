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
    from .io.sequence_reader import read_sequences
    from .io.airr import write_airr

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

    calibration = b_calibration
    if args.calibration and os.path.exists(args.calibration):
        calibration = json.load(open(args.calibration))           # explicit flag overrides bundle

    try:
        ids, seqs, info = read_sequences(args.input, seq_column=args.sequence_column,
                                         id_column=args.id_column)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"error: {e}")
    log(f"read {info['n_read']} sequences ({info['n_dropped']} dropped) as {info['format']}")
    if not seqs:
        raise SystemExit("error: no valid sequences to align")

    preds = predict_reads(model, rs, seqs, device=device, batch_size=args.batch,
                          rerank="learned", v_reader=args.v_reader, calibration=calibration)
    # coordinates are in the canonical (forward) frame -> emit the canonical sequence so
    # they always match it, even for reverse-complemented input reads (with rev_comp flag).
    canon = [canonicalize_sequence(s, p["orientation_id"]) for s, p in zip(seqs, preds)]
    to_stdout = args.output == "-"
    write_provenance = not (args.no_provenance or to_stdout)
    try:
        if not to_stdout:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        write_airr(args.output, ids, canon, preds, locus=locus)
        if write_provenance:
            _write_provenance(args.output + ".run.json", args=args, model_path=model_path,
                              device=device, info=info, ref_desc=ref_desc)
    except (PermissionError, OSError) as e:
        raise SystemExit(
            f"error: cannot write output to {args.output}: {e}\n"
            f"hint: if running in Docker, mount a writable output directory and add "
            f"`--user $(id -u):$(id -g)` so files are written as you.")
    if not to_stdout:
        log(f"wrote {len(preds)} rearrangements -> {args.output}"
            + (f"  (+ {os.path.basename(args.output)}.run.json)" if write_provenance else ""))


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
        print(f"{'id':22s} {'species':8s} {'locus':6s} description")
        for mid, e in MODEL_CATALOG.items():
            print(f"{mid:22s} {e.get('species',''):8s} {e.get('locus',''):6s} {e.get('description','')}")
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

    print(f"[train] resolving reference ...")
    dc, rs, bundle_ref = _resolve_train_reference(args)
    genes = ["v", "j"] + (["d"] if rs.has_d else [])
    locus = args.locus or "IGH"
    print(f"[train] reference: V={len(rs.gene('V').names)} "
          f"{'D=' + str(len(rs.gene('D').names)) + ' ' if rs.has_d else ''}J={len(rs.gene('J').names)}"
          f"  | device={device} preset={args.preset} steps={steps} batch={batch}")

    # model: fine-tune from a base bundle (transfers weights; reference may differ) or fresh
    if args.base_model:
        model, _, _, _, _ = _load_model(args.base_model, device)
        print(f"[train] fine-tuning from base model: {args.base_model}")
    else:
        cfg = DNAlignAIRConfig(d_model=d_model, n_layers=n_layers, nhead=nhead)
        model = DNAlignAIR(cfg).to(device)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    gym = AlignAIRGym([dc], rs, seed=args.seed, allow_curatable=args.allow_curatable)
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=args.lr, batch_size=batch)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model: {n_params/1e6:.2f}M params")

    ckpt_path = os.path.join(out, "checkpoint.pt")
    t0 = time.time()
    step = 0
    while step < steps:
        chunk = min(args.eval_every, steps - step)
        trainer.fit(total_steps=chunk, global_total=steps)
        step += chunk
        ev = trainer.evaluate(n_batches=3)
        call = " ".join(f"{g.upper()}={ev[f'{g}_call']:.2f}" for g in genes)
        print(f"[train] step {step:5d}/{steps}  {time.time()-t0:5.0f}s  "
              f"loss={ev['loss']:.3f} region={ev['region_acc']:.3f}  call: {call}", flush=True)
        torch.save({"model": model.state_dict(), "config": model.config.to_dict(),
                    "step": step}, ckpt_path)            # checkpoint (resume reloads weights)

    # final evaluation -> validation report
    ev = trainer.evaluate(n_batches=8)
    ev_clean = trainer.evaluate(n_batches=8, p=0.0)
    report = {"reference": bundle_ref.get("dataconfigs", ["custom-fasta"]),
              "locus": locus, "steps": steps, "preset": args.preset, "seed": args.seed,
              "n_params": n_params, "wall_seconds": round(time.time() - t0, 1),
              "v_count": len(rs.gene("V").names), "j_count": len(rs.gene("J").names),
              "d_count": len(rs.gene("D").names) if rs.has_d else 0,
              "eval_hard": {g: round(ev[f"{g}_call"], 4) for g in genes},
              "eval_clean": {g: round(ev_clean[f"{g}_call"], 4) for g in genes},
              "region_acc": round(ev["region_acc"], 4), "orientation_acc": round(ev["orient_acc"], 4)}
    _json.dump(report, open(os.path.join(out, "validation_report.json"), "w"), indent=2)

    # save a loadable, self-contained bundle
    bundle_dir = os.path.join(out, "bundle")
    save_dnalignair_bundle(bundle_dir, model=model, locus=locus,
                           dataconfigs=bundle_ref.get("dataconfigs"),
                           reference_set=bundle_ref.get("reference_set"),
                           notes=f"alignair train preset={args.preset} steps={steps} seed={args.seed}")

    # human-readable model card
    ref_desc = (", ".join(bundle_ref["dataconfigs"]) if "dataconfigs" in bundle_ref
                else f"custom FASTA ({locus}, V={report['v_count']}/J={report['j_count']}/D={report['d_count']})")
    card = (f"# AlignAIR model\n\n"
            f"- **Reference**: {ref_desc}\n- **Locus**: {locus}\n- **Preset**: {args.preset}  "
            f"(steps={steps}, params={n_params/1e6:.2f}M, seed={args.seed})\n"
            f"- **Trained**: {report['wall_seconds']}s on {device}\n\n"
            f"## Validation (GenAIRR-simulated)\n\n"
            f"| gene | call acc (hard) | call acc (clean) |\n|---|---|---|\n"
            + "".join(f"| {g.upper()} | {report['eval_hard'][g]:.3f} | {report['eval_clean'][g]:.3f} |\n"
                      for g in genes)
            + f"\nregion_acc={report['region_acc']:.3f}  orientation_acc={report['orientation_acc']:.3f}\n\n"
            f"## Use\n\n```bash\nalignair predict reads.fasta -o out.tsv --model {bundle_dir}\n```\n")
    open(os.path.join(out, "model_card.md"), "w").write(card)

    print(f"\n[train] done -> bundle: {bundle_dir}")
    print(f"[train] validation_report.json + model_card.md written to {out}/")
    print(f"[train] try it: alignair predict reads.fasta -o out.tsv --model {bundle_dir}")


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
    pr.add_argument("--device", default=None, help="cuda|cpu (auto if unset)")
    pr.add_argument("--v-reader", default="learned", choices=["learned", "parasail"],
                    help="V allele reader: learned (default) or parasail (faster+sharper; needs AlignAIR[reader])")
    pr.add_argument("--quiet", action="store_true", help="suppress progress output")
    pr.add_argument("--no-provenance", action="store_true",
                    help="do not write the <output>.run.json provenance sidecar")
    pr.set_defaults(func=cmd_predict)

    va = sub.add_parser("validate-airr", help="validate a rearrangement TSV against the AIRR-C schema")
    va.add_argument("file", help="AIRR rearrangement TSV to validate")
    va.set_defaults(func=cmd_validate_airr)

    dr = sub.add_parser("doctor", help="check the environment (Python, torch+CUDA, GenAIRR, parasail)")
    dr.add_argument("--model", default=None, help="optionally verify a model bundle/checkpoint resolves")
    dr.set_defaults(func=cmd_doctor)

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
    tr.add_argument("--locus", default="IGH")
    tr.add_argument("--allow-curatable", action="store_true",
                    help="permit simulation from references with curatable issues (e.g. custom "
                         "FASTA alleles with no detected anchor); required for some custom references")
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
