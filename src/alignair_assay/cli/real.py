"""``real``: the real-data pilot, from each donor's raw reads to scored, cluster-aware comparisons, resumably."""
from __future__ import annotations

import sys

from alignair_assay.cli.common import add_checkpoint_option, checkpoint_map, comma_list, tool_path
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.reference import DEFAULT_REFERENCE
from alignair_assay.real.preprocess import CAP
from alignair_assay.runners.alignair import DEFAULT_IMAGE, DEFAULT_MODEL
from alignair_assay.runners.base import absolute_path
from alignair_assay.workflow.real import SPLITS, STOP_AFTER, TOOLS, RealConfig, default_checkpoint, run_real


def register(sub) -> None:
    """Add the ``real`` subcommand."""
    p = sub.add_parser("real", help="score the tools on real AIRR-seq reads against each donor's genomic genotype")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--data-root", required=True, help="directory of the PRJNA555323 downloads (never the run dir)")
    p.add_argument("--donors", required=True, type=comma_list, help="comma-separated pilot donors, e.g. SC-23,W-97")
    p.add_argument("--split", choices=SPLITS, default="dev",
                   help="dev for iteration (default); test donors are locked and run only with --split test")
    p.add_argument("--cap", type=int, default=CAP, help=f"preprocessed sequences kept per donor (default {CAP})")
    p.add_argument("--tools", type=comma_list, default=TOOLS,
                   help=f"tools to run (default {','.join(TOOLS)}); alignair-api@LABEL runs the --checkpoint of "
                        "LABEL as a tool of its own, compared with igblast and with the other checkpoints, the "
                        "baseline first: Δ = the second − the first (a label named base given second is warned of)")
    p.add_argument("--threads", type=int, default=8, help="tool and pRESTO threads")
    p.add_argument("--device", default="cpu", help="torch device of alignair-api, e.g. cpu or cuda")
    p.add_argument("--batch-size", type=int, default=64, help="alignair-api reads per batch")
    p.add_argument("--paper-length-filter", action="store_true",
                   help="the paper's multiplex filter (at least 400 nt before primer removal)")
    p.add_argument("--accept-assembly-loss", action="store_true",
                   help="let a run whose assembly loses more pairs than the acceptance line through")
    p.add_argument("--reference", default=DEFAULT_REFERENCE, help="GenAIRR dataconfig of the shared reference")
    p.add_argument("--igblast-dir", help="directory of an unpacked IgBLAST release (igblast)")
    p.add_argument("--alignair-image", help=f"AlignAIR Docker image pinned by digest (default {DEFAULT_IMAGE})")
    p.add_argument("--alignair-model", default=DEFAULT_MODEL, help="AlignAIR registry model id@version (alignair)")
    p.add_argument("--alignair-checkpoint",
                   help="AlignAIR .alignair model file of alignair-api (default: the user's cached file of "
                        "--alignair-model, the image's model)")
    add_checkpoint_option(p, "alignair-api")
    p.add_argument("--stop-after", choices=STOP_AFTER,
                   help="stop after this step: preprocess fetches, genotypes and preprocesses each donor (the "
                        "protocol classifier included) and scores nothing, so it takes any pilot donor, test donors "
                        "too, and needs no tool option")
    p.add_argument("--force", action="store_true",
                   help="proceed in a run directory made for another configuration, redoing what the change affects")
    p.set_defaults(func=run)


def run(args) -> int:
    """Run ``workflow.real.run_real``; 0, or 2 when a donor stopped, a tool run failed or a donor's truth self-check
    failed (each named on stderr; an unchecked self-check is a warning). Refusals exit with status 2 before any
    work."""
    checkpoint = args.alignair_checkpoint or default_checkpoint(args.alignair_model)
    cfg = RealConfig(donors=args.donors, split=args.split, data_root=absolute_path(args.data_root), cap=args.cap,
                     tools=args.tools, threads=args.threads, device=args.device,
                     paper_length_filter=args.paper_length_filter, accept_assembly_loss=args.accept_assembly_loss,
                     reference=args.reference, igblast_dir=tool_path(args.igblast_dir),
                     alignair_image=args.alignair_image, alignair_model=args.alignair_model,
                     alignair_checkpoint=tool_path(checkpoint), batch_size=args.batch_size,
                     stop_after=args.stop_after, checkpoints=checkpoint_map(args.checkpoints))
    layout = RunLayout(absolute_path(args.run_dir))
    summary = run_real(cfg, layout, force=args.force)
    print(f"summary: {layout.real_summary_json()}", flush=True)
    print(f"report: {layout.real_report_md()}", flush=True)
    for stopped in summary["stopped_donors"]:
        print(f"error: {stopped['donor']} stopped at {stopped['step']}: {stopped['reason']}", file=sys.stderr)
    for failed in summary["failed_runs"]:
        print(f"error: {failed['tool']} on {failed['donor']}'s {failed['arm']} arm failed: {failed['error']}"
              + (f" ({failed['note']})" if failed["note"] else ""), file=sys.stderr)
    for donor in summary["oracle_failures"]:
        print(f"error: {donor}'s truth self-check failed: the oracle's known answers are not all right and "
              "supported", file=sys.stderr)
    for donor in summary["oracle_unchecked"]:
        print(f"warning: {donor}'s truth self-check is unchecked: no oracle read is scored (the reads have no "
              "documented known answer, or every one is excluded, e.g. not_captured)", file=sys.stderr)
    return 2 if summary["stopped_donors"] or summary["failed_runs"] or summary["oracle_failures"] else 0
