"""``all``: build suites, run the oracle and two tools, score, compare and write the run report, resumably."""
from __future__ import annotations

from alignair_assay.cli.common import add_checkpoint_option, checkpoint_map, comma_list, tool_path
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.reference import DEFAULT_REFERENCE
from alignair_assay.runners import RUNNERS
from alignair_assay.runners.alignair import DEFAULT_IMAGE, DEFAULT_MODEL
from alignair_assay.runners.base import absolute_path
from alignair_assay.suites.engine import DEFAULT_SEED, SPLITS
from alignair_assay.workflow.orchestrate import RunConfig, run_all

# ``--quick``: the scale of a smoke run (S1 then has 100 cases).
QUICK_SCALE = 0.02


def register(sub) -> None:
    """Add the ``all`` subcommand."""
    p = sub.add_parser("all", help="build, run, score, compare and report in one resumable command")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--suites", required=True, type=comma_list, help="comma-separated suite names, e.g. S1,S10,S11")
    p.add_argument("--tools", required=True, type=comma_list,
                   help=f"the two compared tools, a,b ({', '.join(RUNNERS)}, or a checkpoint variant "
                        "alignair-local@LABEL / alignair-api@LABEL with --checkpoint LABEL=PATH), the baseline first: "
                        "Δ = the second − the first (a label named base given second, or variants of two bases, "
                        "is warned of); the oracle always runs as well")
    p.add_argument("--split", choices=SPLITS, default="dev",
                   help="dev for model iteration (default); test is locked and used only for claims")
    p.add_argument("--reference", default=DEFAULT_REFERENCE, help="GenAIRR dataconfig name")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--scale", type=float, help=f"multiplies every stratum's case count (default 1.0, or "
                                               f"{QUICK_SCALE} with --quick)")
    p.add_argument("--workers", type=int, default=8, help="processes generating strata in parallel")
    p.add_argument("--threads", type=int, default=8, help="tool threads")
    p.add_argument("--igblast-dir", help="directory of an unpacked IgBLAST release (igblast)")
    p.add_argument("--alignair-image", help=f"AlignAIR Docker image (alignair; default {DEFAULT_IMAGE})")
    p.add_argument("--alignair-model", default=DEFAULT_MODEL, help="AlignAIR registry model id@version (alignair)")
    p.add_argument("--alignair-checkpoint", help="AlignAIR .alignair model file (alignair-local)")
    p.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda (alignair-local)")
    add_checkpoint_option(p, "alignair-local or alignair-api")
    p.add_argument("--speed", action="store_true",
                   help="also time each tool on S1's reads (the speed protocol; needs S1 at scale 0.8 or more)")
    p.add_argument("--quick", action="store_true", help=f"a smoke run at scale {QUICK_SCALE} unless --scale is given")
    p.add_argument("--force", action="store_true",
                   help="proceed in a run directory made for another configuration, redoing what the change affects")
    p.set_defaults(func=run)


def run(args) -> int:
    """Run ``workflow.orchestrate.run_all``; its return code (2 when a run failed or the oracle is not perfect).
    The run directory and the tool paths (``--igblast-dir``, ``--alignair-checkpoint``, each ``--checkpoint``) are
    made absolute first, so the manifest never records a path relative to where the command was started; a
    ``--checkpoint`` label given twice is refused."""
    scale = args.scale if args.scale is not None else QUICK_SCALE if args.quick else 1.0
    cfg = RunConfig(suites=args.suites, tools=args.tools, split=args.split, reference=args.reference, seed=args.seed,
                    scale=scale, workers=args.workers, threads=args.threads, igblast_dir=tool_path(args.igblast_dir),
                    alignair_image=args.alignair_image, alignair_model=args.alignair_model,
                    alignair_checkpoint=tool_path(args.alignair_checkpoint), device=args.device, speed=args.speed,
                    checkpoints=checkpoint_map(args.checkpoints))
    return run_all(cfg, RunLayout(absolute_path(args.run_dir)), force=args.force)
