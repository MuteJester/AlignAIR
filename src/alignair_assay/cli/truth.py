"""``truth``: build observable truth from bench cases and print a summary of what could not be scored."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from alignair_assay.cli.common import DEFAULT_REFERENCE
from alignair_assay.core.jsonl import read_jsonl
from alignair_assay.core.prediction import GENES
from alignair_assay.core.provenance import genairr_provenance
from alignair_assay.core.reference import Reference
from alignair_assay.scoring.pipeline import check_unique, simulator_disagreements
from alignair_assay.truth.bench import truth_from_bench_case
from alignair_assay.truth.io import TRUTH_KIND, TRUTH_VERSION, write_truth_jsonl

_MAX_LISTED_WARNINGS = 20


def register(sub) -> None:
    """Add the ``truth`` subcommand."""
    p = sub.add_parser("truth", help="build observable truth from bench cases")
    p.add_argument("--bench-cases", required=True, help="bench case JSONL (.gz allowed)")
    p.add_argument("--reference", default=DEFAULT_REFERENCE, help="GenAIRR dataconfig name")
    p.add_argument("--out", required=True, help="truth JSONL to write (.gz compresses)")
    p.set_defaults(func=run)


def run(args) -> int:
    """Write the truth file (header: kind, version, reference, its digest, GenAIRR provenance) and print counts of
    excluded cases, warnings, unscorable genes and simulator allele-set disagreements."""
    reference = Reference.from_dataconfig(args.reference)
    truths = [truth_from_bench_case(case, reference) for case in read_jsonl(args.bench_cases)]
    check_unique(truths, args.bench_cases)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = {"kind": TRUTH_KIND, "version": TRUTH_VERSION, "reference": args.reference,
              "reference_digest": reference.digest(), **genairr_provenance()}
    write_truth_jsonl(truths, out, header=header)

    excluded = Counter(t.excluded for t in truths if t.excluded is not None)
    with_warnings = [t for t in truths if t.warnings]
    unscorable = Counter(g for t in truths for g in t.unscorable)
    reasons = ", ".join(f"{k}: {v}" for k, v in sorted(excluded.items()))
    print(f"cases: {len(truths)}")
    print(f"excluded: {sum(excluded.values())}" + (f" ({reasons})" if reasons else ""))
    print(f"cases with warnings: {len(with_warnings)}")
    print("unscorable genes: " + " ".join(f"{g}={unscorable[g]}" for g in GENES))
    disagreements = simulator_disagreements(truths)
    print("simulator allele-set disagreements (non-excluded cases): "
          + " ".join(f"{g}={disagreements[g]}" for g in GENES))
    for t in with_warnings[:_MAX_LISTED_WARNINGS]:
        for w in t.warnings:
            print(f"  {t.case_id}: {w}")
    if len(with_warnings) > _MAX_LISTED_WARNINGS:
        print(f"  ... and {len(with_warnings) - _MAX_LISTED_WARNINGS} more cases with warnings")
    print(f"wrote {out}")
    return 0
