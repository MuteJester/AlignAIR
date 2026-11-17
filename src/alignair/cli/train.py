"""``alignair train`` — train a custom AlignAIR model on a built-in GenAIRR dataconfig OR your own
V/D/J germline FASTAs, and export a safe (pickle-free) distributable bundle."""
from __future__ import annotations

import json
import os

from ..errors import ConfigurationError, report_error
from ._errors import fail
from ._options import add_device


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
    p.add_argument("--lr", type=float, default=3e-4, help="initial learning rate (default 3e-4)")
    p.add_argument("--batch-size", type=int, default=None, help="override the preset's batch size")
    p.add_argument("--grad-clip", type=float, default=None, help="clip the global gradient norm")
    p.add_argument("--val-every", type=int, default=None, help="override the preset's validation interval")
    p.add_argument("--seed", type=int, default=0, help="seed of the training data and initial weights (default 0)")
    p.add_argument("--data-workers", type=int, default=0,
                   help="generate training data in this many worker processes (at most one per curriculum "
                        "stream: 28 for the default mix, 7 components of 4 streams); the batches are identical "
                        "to --data-workers 0 (default: 0)")
    p.add_argument("--plan", action="store_true",
                   help="validate the reference/config and print the plan WITHOUT training")
    add_device(p)
    p.set_defaults(func=run)


# The flag that sets each field of the training reference (``genairr.custom.build_dataconfigs``).
_REFERENCE_FLAGS = {"dataconfig": "--dataconfig", "v_fasta": "--v-fasta", "d_fasta": "--d-fasta",
                    "j_fasta": "--j-fasta", "chain_type": "--chain-type"}


def training_plan(dcs, *, steps: int, batch_size: int) -> dict:
    """A pre-flight plan (no training): reference composition, model parameter count, and the resolved
    schedule — so a user can sanity-check a custom reference / preset before committing GPU time."""
    from ..core import prepare_model
    from ..reference.reference_set import ReferenceSet
    ref = ReferenceSet.from_dataconfigs(*dcs)
    setup = prepare_model(ref)
    cfg = setup.config
    params = sum(p.numel() for p in setup.build_model().parameters())
    return {
        "loci": list(ref.locus_names()),
        "alleles": {G: len(ref.gene(G)) for G in ("V", "D", "J") if G in ref.genes},
        "has_d": ref.has_d, "max_seq_length": cfg.max_seq_length,
        "model_parameters": params, "steps": steps, "batch_size": batch_size,
    }


def run(args) -> int:
    from ..genairr.custom import build_dataconfigs
    from ..train.bundle import export_bundle

    try:
        dcs, report = build_dataconfigs(dataconfig=args.dataconfig, v_fasta=args.v_fasta,
                                        d_fasta=args.d_fasta, j_fasta=args.j_fasta,
                                        chain_type=args.chain_type)
    except ValueError as e:                    # ours (a ConfigurationError), or GenAIRR's on a FASTA it cannot build
        flag = _REFERENCE_FLAGS.get(e.context.get("field")) if isinstance(e, ConfigurationError) else None
        if flag and e.hint is None:
            e.hint = (f"{flag} sets it. Name built-in references with --dataconfig, or a custom one with --v-fasta, "
                      "--j-fasta and --chain-type (and --d-fasta for a locus with a D gene).")
        return fail(e, error_type=ConfigurationError, context={"stage": "training_reference"})

    from ..train.config import TrainingConfig
    from ..train.workflow import run_training
    overrides = {"lr": args.lr, "grad_clip": args.grad_clip, "seed": args.seed, "data_workers": args.data_workers,
                 "device": args.device}
    if args.steps is not None:
        overrides["steps"] = args.steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.val_every is not None:
        overrides["val_every"] = args.val_every
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
        return report_error(ConfigurationError(
            f"bundle already exists: {bundle_dir}. Refusing to overwrite a possibly-published bundle — "
            f"choose a new --out or pass --overwrite. (No training was run.)",
            context={"path": bundle_dir, "stage": "training_preflight"}))

    run = run_training(cfg, output_dir=args.out)

    checkpoint = run.best_model_path or run.model_path
    sources = {"v_fasta": args.v_fasta, "d_fasta": args.d_fasta, "j_fasta": args.j_fasta} if args.v_fasta else None
    # No `training=` override: export_bundle reads the full provenance from the checkpoint (actual
    # steps + curriculum + effective per-locus SHM caps); a dict restated here would drop the fields it does not list,
    # such as `train_args`.
    bundle = export_bundle(checkpoint, dcs, os.path.join(args.out, "bundle"),
                           description="AlignAIR model (custom reference)" if args.v_fasta else "",
                           sources=sources, report=report, overwrite=args.overwrite)
    print(f"trained -> {run.model_path}\nexported pickle-free bundle -> {bundle}")
    return 0
