"""``score``: score one tool's AIRR TSV against truth and write its scores and report."""
from __future__ import annotations

from pathlib import Path

from alignair_assay.cli.common import file_name, print_matching
from alignair_assay.core.reference import DEFAULT_REFERENCE, Reference
from alignair_assay.scoring.pipeline import load_truth, write_scores
from alignair_assay.workflow.scoring import read_genotypes, score_file, score_report, write_report


def register(sub) -> None:
    """Add the ``score`` subcommand."""
    p = sub.add_parser("score", help="score one tool's AIRR TSV against truth")
    p.add_argument("--truth", required=True)
    p.add_argument("--predictions", required=True, help="AIRR rearrangement TSV")
    p.add_argument("--tool", required=True, type=file_name, help="tool name (used in output file names)")
    p.add_argument("--reference", default=DEFAULT_REFERENCE, help="GenAIRR dataconfig name")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--donors", help="the suite's donors.json: also score each read against its donor's genotype")
    p.set_defaults(func=run)


def run(args) -> int:
    """Write ``<tool>.scores.jsonl.gz``, ``<tool>.report.json`` and ``<tool>.report.md`` into ``--out-dir``. With
    ``--donors``, each read with a subject is also scored against its donor's genotype (the genotype metrics), and
    the report holds the excess-alleles table; a subject the donors do not have is an input error."""
    reference = Reference.from_dataconfig(args.reference)
    header, truths = load_truth(args.truth, args.reference, reference)
    genotypes = read_genotypes(args.donors) if args.donors else None
    scored = score_file(args.tool, truths, args.predictions, reference, genotypes=genotypes)
    print_matching(args.tool, scored.matching)
    inputs = {"truth": args.truth, "predictions": args.predictions} | ({"donors": args.donors} if args.donors else {})
    report = score_report(scored, truths, header, command="score", inputs=inputs, reference_name=args.reference)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = (out / f"{args.tool}.scores.jsonl.gz", out / f"{args.tool}.report.json", out / f"{args.tool}.report.md")
    write_scores(scored.scores, paths[0])
    write_report(report, paths[1], paths[2])
    print("wrote " + ", ".join(map(str, paths)))
    return 0
