"""``audit``: the readiness gate for model comparisons (``workflow.audit``): self-checks, reproducibility, null
comparisons of one checkpoint with itself and IgBLAST's bands, over a stored ``all`` and ``real`` run."""
from __future__ import annotations

from alignair_assay.cli.common import checkpoint_spec, tool_path
from alignair_assay.core.layout import RunLayout
from alignair_assay.runners.base import absolute_path


def register(sub) -> None:
    """Add the ``audit`` subcommand."""
    p = sub.add_parser("audit", help="the readiness gate: prove the truth checks itself, outputs reproduce, a "
                                     "checkpoint against itself differs by nothing and IgBLAST lands in its bands")
    p.add_argument("--sim-run", required=True, help="a run directory made by all (read only: its oracle and IgBLAST "
                                                    "numbers, e.g. the test-split claim run)")
    p.add_argument("--real-run", required=True, help="a run directory made by real on the dev donors (read only)")
    p.add_argument("--data-root", required=True, help="directory of the PRJNA555323 downloads")
    p.add_argument("--checkpoint", required=True, type=checkpoint_spec, metavar="base=PATH",
                   help="the audited .alignair checkpoint, run against itself in the null comparisons")
    p.add_argument("--device", default="cpu", help="torch device of the null comparisons, e.g. cpu or cuda")
    p.add_argument("--skip-speed", action="store_true", help="do not time IgBLAST (the informational speed band)")
    p.add_argument("--igblast-dir", help="the IgBLAST release to time (default: the one --sim-run recorded)")
    p.add_argument("--workers", type=int, default=8, help="processes building suites")
    p.add_argument("--threads", type=int, default=8, help="tool threads")
    p.add_argument("--out", required=True, help="the audit's directory (its audit/ holds audit.json, audit.md and "
                                                "the scratch runs); never inside --sim-run or --real-run")
    p.set_defaults(func=run)


def run(args) -> int:
    """Run ``workflow.audit.run_audit`` (which prints each check as it finishes), print the verdict and name the
    record; 0 when the audit passes (no check failed, and one passed), else 2. Refusals (another checkpoint label, a
    missing input, an output directory overlapping a stored run) exit with status 2 before any work."""
    # Imported here so the other commands never load the audit's workflows.
    from alignair_assay.workflow import audit

    label, checkpoint = args.checkpoint
    if label != audit.CHECKPOINT_LABEL:
        raise ValueError(f"the audit takes one checkpoint, --checkpoint {audit.CHECKPOINT_LABEL}=PATH; got the "
                         f"label {label!r}")

    cfg = audit.AuditConfig(sim_run=absolute_path(args.sim_run), real_run=absolute_path(args.real_run),
                            data_root=absolute_path(args.data_root), checkpoint=checkpoint, device=args.device,
                            skip_speed=args.skip_speed, igblast_dir=tool_path(args.igblast_dir),
                            workers=args.workers, threads=args.threads)
    layout = RunLayout(absolute_path(args.out))
    record = audit.run_audit(cfg, layout)
    print(record["verdict"]["line"])
    print(f"audit: {layout.audit_md} ({layout.audit_json})")
    return 0 if record["verdict"]["status"] == "pass" else 2
