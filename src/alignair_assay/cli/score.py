"""``score``: score one tool's AIRR TSV against truth and write its scores and report."""
from __future__ import annotations

from pathlib import Path

from alignair_assay.cli.common import DEFAULT_REFERENCE, file_name, print_matching, provenance, write_report
from alignair_assay.core.reference import Reference
from alignair_assay.report import build_score_report
from alignair_assay.scoring.pipeline import load_truth, score_tool, simulator_disagreements, write_scores
from alignair_assay.scoring.slices import slices
from alignair_assay.stats import summarize


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
    scores, matching = score_tool(truths, args.predictions, reference)
    print_matching(args.tool, matching)
    prov = provenance("score", {"truth": args.truth, "predictions": args.predictions},
                      tool=args.tool, reference=args.reference, reference_digest=header["reference_digest"],
                      truth_header=header, matching=matching,
                      simulator_disagreements=simulator_disagreements(truths))
    report = build_score_report(args.tool, summarize(scores, list(scores)), slices(scores, truths), prov)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = (out / f"{args.tool}.scores.jsonl.gz", out / f"{args.tool}.report.json", out / f"{args.tool}.report.md")
    write_scores(scores, paths[0])
    write_report(report, paths[1], paths[2])
    print("wrote " + ", ".join(map(str, paths)))
    return 0
