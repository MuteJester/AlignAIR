"""``compare``: paired comparison of two tools' AIRR TSVs on the same truth."""
from __future__ import annotations

from pathlib import Path

from alignair_assay.cli.common import DEFAULT_REFERENCE, file_name, print_matching, provenance, write_report
from alignair_assay.core.reference import Reference
from alignair_assay.metrics import is_informational, lower_is_better
from alignair_assay.report import build_comparison_report, headline_table
from alignair_assay.scoring.pipeline import load_truth, score_tool, simulator_disagreements
from alignair_assay.scoring.slices import slices
from alignair_assay.stats import compare

STATISTICAL_TESTS = ("exact McNemar for 0/1 metrics, paired z-test otherwise, p = 1 below 10 pairs; "
                     "Holm across the family; normal-theory CIs (95% and simultaneous Bonferroni)")


def register(sub) -> None:
    """Add the ``compare`` subcommand (``main`` checks that the two names differ)."""
    p = sub.add_parser("compare", help="paired comparison of two tools' AIRR TSVs")
    p.add_argument("--truth", required=True)
    p.add_argument("--a", required=True, help="AIRR TSV of tool a")
    p.add_argument("--a-name", required=True, type=file_name)
    p.add_argument("--b", required=True, help="AIRR TSV of tool b")
    p.add_argument("--b-name", required=True, type=file_name)
    p.add_argument("--reference", default=DEFAULT_REFERENCE, help="GenAIRR dataconfig name")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--alpha", type=float, default=0.05, help="family-wise error rate of the Holm verdicts")
    p.set_defaults(func=run)


def run(args) -> int:
    """Compare every non-informational metric either tool produced (Holm across that family), write
    ``comparison.json`` and ``comparison.md`` into ``--out-dir`` and print the headline table."""
    reference = Reference.from_dataconfig(args.reference)
    header, truths = load_truth(args.truth, args.reference, reference)
    a, match_a = score_tool(truths, args.a, reference)
    b, match_b = score_tool(truths, args.b, reference)
    print_matching(args.a_name, match_a)
    print_matching(args.b_name, match_b)
    seen = {name for scores in (a, b) for case in scores.values() for name in case}
    family = sorted(name for name in seen if not is_informational(name))
    comparison = compare(a, b, family, alpha=args.alpha,
                         lower_is_better={name for name in family if lower_is_better(name)})
    prov = provenance("compare", {"truth": args.truth, "a": args.a, "b": args.b},
                      a=args.a_name, b=args.b_name, reference=args.reference,
                      reference_digest=header["reference_digest"], truth_header=header, alpha=args.alpha,
                      tests=STATISTICAL_TESTS, family_size=len(family),
                      matching={"a": match_a, "b": match_b},
                      simulator_disagreements=simulator_disagreements(truths))
    report = build_comparison_report(args.a_name, args.b_name, comparison, slices(a, truths), slices(b, truths),
                                     prov)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = (out / "comparison.json", out / "comparison.md")
    write_report(report, *paths)
    print(f"compared {len(family)} metrics; headline:")
    print(headline_table(report), end="")
    print("wrote " + ", ".join(map(str, paths)))
    return 0
