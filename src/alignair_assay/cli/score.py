"""``score``: score one tool's AIRR TSV against truth and write its scores and report."""
from __future__ import annotations

from pathlib import Path

from alignair_assay.cli.common import file_name, print_matching
from alignair_assay.core.reference import DEFAULT_REFERENCE, Reference
from alignair_assay.scoring.pipeline import load_truth, score_tool, write_scores
from alignair_assay.workflow.scoring import ToolScores, score_report, write_report


def register(sub) -> None:
    """Add the ``score`` subcommand."""
    p = sub.add_parser("score", help="score one tool's AIRR TSV against truth")
    p.add_argument("--truth", required=True)
    p.add_argument("--predictions", required=True, help="AIRR rearrangement TSV")
    p.add_argument("--tool", required=True, type=file_name, help="tool name (used in output file names)")
    p.add_argument("--reference", default=DEFAULT_REFERENCE, help="GenAIRR dataconfig name")
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=run)


def run(args) -> int:
    """Write ``<tool>.scores.jsonl.gz``, ``<tool>.report.json`` and ``<tool>.report.md`` into ``--out-dir``."""
    reference = Reference.from_dataconfig(args.reference)
    header, truths = load_truth(args.truth, args.reference, reference)
    scored = ToolScores(args.tool, *score_tool(truths, args.predictions, reference))
    print_matching(args.tool, scored.matching)
    report = score_report(scored, truths, header, command="score",
                          inputs={"truth": args.truth, "predictions": args.predictions}, reference_name=args.reference)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = (out / f"{args.tool}.scores.jsonl.gz", out / f"{args.tool}.report.json", out / f"{args.tool}.report.md")
    write_scores(scored.scores, paths[0])
    write_report(report, paths[1], paths[2])
    print("wrote " + ", ".join(map(str, paths)))
    return 0
