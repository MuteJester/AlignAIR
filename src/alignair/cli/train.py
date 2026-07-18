"""``alignair train`` — train a custom AlignAIR model on a built-in GenAIRR dataconfig OR your own
V/D/J germline FASTAs, and export a safe (pickle-free) distributable bundle."""
from __future__ import annotations

import json
import os


def register(sub) -> None:
    p = sub.add_parser("train", help="train a model on a built-in dataconfig or custom V/D/J FASTAs")
    p.add_argument("--dataconfig", nargs="+", default=None,
                   help="built-in GenAIRR dataconfig(s) (1 => single-chain, N => multi-chain); "
                        "or use --v-fasta/--j-fasta/--chain-type for a custom reference")
    p.add_argument("--v-fasta", default=None, help="custom V germline FASTA (with --j-fasta [--d-fasta])")
    p.add_argument("--d-fasta", default=None, help="custom D germline FASTA (heavy / D-bearing loci)")
    p.add_argument("--j-fasta", default=None, help="custom J germline FASTA")
    p.add_argument("--chain-type", default=None,
                   choices=["BCR_HEAVY", "BCR_LIGHT_KAPPA", "BCR_LIGHT_LAMBDA",
                            "TCR_ALPHA", "TCR_BETA", "TCR_GAMMA", "TCR_DELTA"],
                   help="GenAIRR chain type for a custom reference")
    p.add_argument("--out", required=True, help="output run directory (checkpoints + bundle/)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing bundle/ in --out (default: refuse, to protect a published bundle)")
    p.add_argument("--preset", choices=["quick", "desktop", "full"], default="desktop",
                   help="resource-tuned defaults for steps/batch/validation (default: desktop)")
    p.add_argument("--steps", type=int, default=None, help="override the preset's step count")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=None, help="override the preset's batch size")
    p.add_argument("--grad-clip", type=float, default=None, help="clip the global gradient norm")
    p.add_argument("--val-every", type=int, default=None, help="override the preset's validation interval")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plan", action="store_true",
                   help="validate the reference/config and print the plan WITHOUT training")
    p.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    p.set_defaults(func=run)


def run(args) -> int:
    from ..train.build import build_dataconfigs, export_bundle, training_plan
    from ..train.guards import TrainingConfigError

    try:
        dcs, report = build_dataconfigs(dataconfig=args.dataconfig, v_fasta=args.v_fasta,
                                        d_fasta=args.d_fasta, j_fasta=args.j_fasta,
                                        chain_type=args.chain_type)
    except ValueError as e:
        print(str(e))
        return 1

    from ..aligner import TrainingConfig, run_training
    overrides = {"lr": args.lr, "grad_clip": args.grad_clip, "seed": args.seed}
    if args.steps is not None:
        overrides["steps"] = args.steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.val_every is not None:
        overrides["val_every"] = args.val_every
    if args.device:
        overrides["device"] = args.device
    cfg = TrainingConfig.from_genairr(*dcs, preset=args.preset, **overrides)

    plan = training_plan(dcs, steps=cfg.steps, batch_size=cfg.batch_size)
    if report and report.warnings:
        print(f"reference build warnings ({len(report.warnings)}): {report.warnings[:5]}")
    print("training plan:\n" + json.dumps(plan, indent=2))
    if args.plan:
        return 0                         # --plan: validate + report only, no training

    # fail closed BEFORE spending training time: don't train for minutes/hours only to refuse the export
    bundle_dir = os.path.join(args.out, "bundle")
    if os.path.exists(bundle_dir) and not args.overwrite:
        print(f"bundle already exists: {bundle_dir}. Refusing to overwrite a possibly-published bundle — "
              f"choose a new --out or pass --overwrite. (No training was run.)")
        return 1

    try:
        run = run_training(cfg, output_dir=args.out)
    except TrainingConfigError as e:
        print(str(e))
        return 1

    checkpoint = run.best_model_path or run.model_path
    sources = {"v_fasta": args.v_fasta, "d_fasta": args.d_fasta, "j_fasta": args.j_fasta} if args.v_fasta else None
    # No `training=` override: export_bundle reads the full provenance from the checkpoint (actual
    # steps + curriculum + effective per-locus SHM caps). Restating a reduced dict here is what used
    # to drop `train_args` from every distributable bundle.
    bundle = export_bundle(checkpoint, dcs, os.path.join(args.out, "bundle"),
                           description="AlignAIR model (custom reference)" if args.v_fasta else "",
                           sources=sources, report=report, overwrite=args.overwrite)
    print(f"trained -> {run.model_path}\nexported pickle-free bundle -> {bundle}")
    return 0
