"""``compare``: paired comparison of two tools' AIRR TSVs on the same truth."""
from __future__ import annotations

from pathlib import Path

from alignair_assay.cli.common import file_name, print_matching
from alignair_assay.core.reference import DEFAULT_REFERENCE, Reference
from alignair_assay.report import headline_table
from alignair_assay.scoring.pipeline import load_truth
from alignair_assay.workflow.scoring import comparison_report, read_genotypes, score_file, write_report


def register(sub) -> None:
    """Add the ``compare`` subcommand."""
    p = sub.add_parser("compare", help="paired comparison of two tools' AIRR TSVs")
    p.add_argument("--truth", required=True)
    p.add_argument("--a", required=True, help="AIRR TSV of tool a")
    p.add_argument("--a-name", required=True, type=file_name)
    p.add_argument("--b", required=True, help="AIRR TSV of tool b")
    p.add_argument("--b-name", required=True, type=file_name)
    p.add_argument("--reference", default=DEFAULT_REFERENCE, help="GenAIRR dataconfig name")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--alpha", type=float, default=0.05, help="family-wise error rate of the Holm verdicts")
    p.add_argument("--donors", help="the suite's donors.json: also compare the genotype metrics")
    p.set_defaults(func=run)


def run(args) -> int:
    """Compare every non-informational metric either tool produced (Holm across that family), write
    ``comparison.json`` and ``comparison.md`` into ``--out-dir`` and print the headline table. With ``--donors``,
    both tools are also scored against each read's donor genotype, so the genotype metrics join the family. Two
    tools with the same name are an input error (their output files would collide)."""
    if args.a_name == args.b_name:
        raise ValueError("--a-name and --b-name must differ")
    reference = Reference.from_dataconfig(args.reference)
    header, truths = load_truth(args.truth, args.reference, reference)
    genotypes = read_genotypes(args.donors) if args.donors else None
    a = score_file(args.a_name, truths, args.a, reference, genotypes=genotypes)
    b = score_file(args.b_name, truths, args.b, reference, genotypes=genotypes)
    print_matching(a.tool, a.matching)
    print_matching(b.tool, b.matching)
    inputs = {"truth": args.truth, "a": args.a, "b": args.b} | ({"donors": args.donors} if args.donors else {})
    report = comparison_report(a, b, truths, header, command="compare", inputs=inputs,
                               reference_name=args.reference, alpha=args.alpha)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = (out / "comparison.json", out / "comparison.md")
    write_report(report, *paths)
    print(f"compared {report['provenance']['family_size']} metrics; headline:")
    print(headline_table(report), end="")
    print("wrote " + ", ".join(map(str, paths)))
    return 0
