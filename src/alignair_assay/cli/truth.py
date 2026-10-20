"""``truth``: build observable truth from bench cases and print a summary of what could not be scored."""
from __future__ import annotations

from pathlib import Path

from alignair_assay.core.jsonl import read_jsonl
from alignair_assay.core.provenance import genairr_provenance
from alignair_assay.core.reference import DEFAULT_REFERENCE, Reference
from alignair_assay.scoring.pipeline import check_unique
from alignair_assay.truth.bench import truth_from_bench_case
from alignair_assay.truth.io import TRUTH_KIND, TRUTH_VERSION, write_truth_jsonl
from alignair_assay.truth.summary import truth_summary

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
    excluded cases, warnings, unscorable genes, simulator call lists that differ from the generative allele,
    simulator re-calls and primary fallbacks."""
    reference = Reference.from_dataconfig(args.reference)
    truths = [truth_from_bench_case(case, reference) for case in read_jsonl(args.bench_cases)]
    check_unique(truths, args.bench_cases)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = {"kind": TRUTH_KIND, "version": TRUTH_VERSION, "reference": args.reference,
              "reference_digest": reference.digest(), **genairr_provenance()}
    write_truth_jsonl(truths, out, header=header)
    _print_truth_summary(truth_summary(truths))
    print(f"wrote {out}")
    return 0


def _print_truth_summary(summary: dict) -> None:
    """Print a ``truth.summary.truth_summary``: the counts, then the warnings of the first cases that have any."""
    excluded = summary["excluded"]
    reasons = ", ".join(f"{k}: {v}" for k, v in excluded.items())
    warnings = summary["warnings"]
    print(f"cases: {summary['cases']}")
    print(f"excluded: {sum(excluded.values())}" + (f" ({reasons})" if reasons else ""))
    print(f"cases with warnings: {len(warnings)}")
    print("unscorable genes: " + " ".join(f"{g}={n}" for g, n in summary["unscorable"].items()))
    calls = summary["simulator_calls"]
    print("simulator call list differs from the generative set (non-excluded cases): "
          + " ".join(f"{g}={n['call_list_differs']}" for g, n in calls.items()))
    print("simulator first call is not the generative allele (non-excluded cases): "
          + " ".join(f"{g}={n['recalled']}" for g, n in calls.items()))
    print("primary from the simulator's call, no generative allele (non-excluded cases): "
          + " ".join(f"{g}={n}" for g, n in summary["primary_fallbacks"].items()))
    for case_id, case_warnings in list(warnings.items())[:_MAX_LISTED_WARNINGS]:
        for w in case_warnings:
            print(f"  {case_id}: {w}")
    if len(warnings) > _MAX_LISTED_WARNINGS:
        print(f"  ... and {len(warnings) - _MAX_LISTED_WARNINGS} more cases with warnings")
