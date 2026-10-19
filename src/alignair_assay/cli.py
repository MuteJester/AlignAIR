"""Command-line entry point: build observable truth, score one tool, compare two tools.

  python -m alignair_assay truth --bench-cases CASES.jsonl[.gz] --reference HUMAN_IGH_OGRDB --out TRUTH.jsonl.gz
  python -m alignair_assay score --truth TRUTH --predictions AIRR.tsv --tool NAME --out-dir DIR
  python -m alignair_assay compare --truth TRUTH --a A.tsv --a-name A --b B.tsv --b-name B --out-dir DIR

Predictions are matched to truth by AIRR ``sequence_id`` == truth ``case_id``. Excluded truths are not scored and
their predictions are ignored; a truth case without a prediction is scored as missing (never dropped).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from alignair_assay.ingest.airr import GENES, load_airr_tsv
from alignair_assay.metrics import CaseScores, is_informational, lower_is_better
from alignair_assay.metrics.case import MIN_OBSERVED, score_case
from alignair_assay.reference import Reference
from alignair_assay.report import build_comparison_report, build_score_report, headline_table, to_markdown
from alignair_assay.stats import by_slice, compare, summarize
from alignair_assay.truth.build import ObservableTruth, truth_from_bench_case
from alignair_assay.truth.io import read_truth_jsonl, write_truth_jsonl

DEFAULT_REFERENCE = "HUMAN_IGH_OGRDB"
SLICE_DIMENSIONS = ("stratum", "orientation", "d_len_bin")
_D_BINS = ((5, "<5"), (8, "5-7"), (10, "8-9"), (12, "10-11"), (15, "12-14"))
_MAX_LISTED_WARNINGS = 20
STATISTICAL_TESTS = ("exact McNemar for 0/1 metrics, paired z-test otherwise, p = 1 below 10 pairs; "
                     "Holm across the family; normal-theory CIs (95% and simultaneous Bonferroni)")


def d_len_bin(length: int | None) -> str:
    """Slice label for an observed D length; ``absent`` when the read shows no D."""
    if length is None:
        return "absent"
    return next((label for upper, label in _D_BINS if length < upper), ">=15")


def slice_labels(truth: ObservableTruth) -> dict[str, str]:
    """The case's label in every slice dimension, derived from its truth."""
    d = truth.segments.get("d")
    return {"stratum": truth.stratum, "orientation": str(truth.orientation),
            "d_len_bin": d_len_bin(d.observed_length if d is not None else None)}


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "compare" and args.a_name == args.b_name:
        parser.error("--a-name and --b-name must differ")
    return args.func(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m alignair_assay",
                                     description="Observable-truth assay for immune-receptor aligners")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("truth", help="build observable truth from bench cases")
    p.add_argument("--bench-cases", required=True, help="bench case JSONL (.gz allowed)")
    p.add_argument("--reference", default=DEFAULT_REFERENCE, help="GenAIRR dataconfig name")
    p.add_argument("--out", required=True, help="truth JSONL to write (.gz compresses)")
    p.set_defaults(func=_cmd_truth)

    p = sub.add_parser("score", help="score one tool's AIRR TSV against truth")
    p.add_argument("--truth", required=True)
    p.add_argument("--predictions", required=True, help="AIRR rearrangement TSV")
    p.add_argument("--tool", required=True, type=_name, help="tool name (used in output file names)")
    p.add_argument("--reference", default=DEFAULT_REFERENCE, help="GenAIRR dataconfig name")
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=_cmd_score)

    p = sub.add_parser("compare", help="paired comparison of two tools' AIRR TSVs")
    p.add_argument("--truth", required=True)
    p.add_argument("--a", required=True, help="AIRR TSV of tool a")
    p.add_argument("--a-name", required=True, type=_name)
    p.add_argument("--b", required=True, help="AIRR TSV of tool b")
    p.add_argument("--b-name", required=True, type=_name)
    p.add_argument("--reference", default=DEFAULT_REFERENCE, help="GenAIRR dataconfig name")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--alpha", type=float, default=0.05, help="family-wise error rate of the Holm verdicts")
    p.set_defaults(func=_cmd_compare)
    return parser


def _name(value: str) -> str:
    if not value or Path(value).name != value:
        raise argparse.ArgumentTypeError(f"not a plain file-name component: {value!r}")
    return value


def _cmd_truth(args) -> int:
    reference = Reference.from_dataconfig(args.reference)
    truths = [truth_from_bench_case(case, reference) for case in _read_jsonl(args.bench_cases)]
    _check_unique(truths, args.bench_cases)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_truth_jsonl(truths, out)

    excluded = Counter(t.excluded for t in truths if t.excluded is not None)
    with_warnings = [t for t in truths if t.warnings]
    unscorable = Counter(g for t in truths for g in t.unscorable)
    reasons = ", ".join(f"{k}: {v}" for k, v in sorted(excluded.items()))
    print(f"cases: {len(truths)}")
    print(f"excluded: {sum(excluded.values())}" + (f" ({reasons})" if reasons else ""))
    print(f"cases with warnings: {len(with_warnings)}")
    print("unscorable genes: " + " ".join(f"{g}={unscorable[g]}" for g in GENES))
    for t in with_warnings[:_MAX_LISTED_WARNINGS]:
        for w in t.warnings:
            print(f"  {t.case_id}: {w}")
    if len(with_warnings) > _MAX_LISTED_WARNINGS:
        print(f"  ... and {len(with_warnings) - _MAX_LISTED_WARNINGS} more cases with warnings")
    print(f"wrote {out}")
    return 0


def _cmd_score(args) -> int:
    truths = _load_truth(args.truth)
    reference = Reference.from_dataconfig(args.reference)
    scores, matching = _score_tool(truths, args.predictions, reference)
    _print_matching(args.tool, matching)
    provenance = _provenance("score", {"truth": args.truth, "predictions": args.predictions},
                             tool=args.tool, reference=args.reference, matching=matching)
    report = build_score_report(args.tool, summarize(scores, list(scores)), _slices(scores, truths), provenance)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = (out / f"{args.tool}.scores.jsonl.gz", out / f"{args.tool}.report.json", out / f"{args.tool}.report.md")
    _write_scores(scores, paths[0])
    _write_report(report, paths[1], paths[2])
    print("wrote " + ", ".join(map(str, paths)))
    return 0


def _cmd_compare(args) -> int:
    truths = _load_truth(args.truth)
    reference = Reference.from_dataconfig(args.reference)
    a, match_a = _score_tool(truths, args.a, reference)
    b, match_b = _score_tool(truths, args.b, reference)
    _print_matching(args.a_name, match_a)
    _print_matching(args.b_name, match_b)
    seen = {name for scores in (a, b) for case in scores.values() for name in case}
    family = sorted(name for name in seen if not is_informational(name))
    comparison = compare(a, b, family, alpha=args.alpha,
                         lower_is_better={name for name in family if lower_is_better(name)})
    provenance = _provenance("compare", {"truth": args.truth, "a": args.a, "b": args.b},
                             a=args.a_name, b=args.b_name, reference=args.reference, alpha=args.alpha,
                             tests=STATISTICAL_TESTS, family_size=len(family),
                             matching={"a": match_a, "b": match_b})
    report = build_comparison_report(args.a_name, args.b_name, comparison, _slices(a, truths), _slices(b, truths),
                                     provenance)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = (out / "comparison.json", out / "comparison.md")
    _write_report(report, *paths)
    print(f"compared {len(family)} metrics; headline:")
    print(headline_table(report), end="")
    print("wrote " + ", ".join(map(str, paths)))
    return 0


def _score_tool(truths: list[ObservableTruth], predictions_path, reference: Reference
                ) -> tuple[dict[str, CaseScores], dict]:
    """Per-case scores of one tool over the non-excluded truths, plus prediction/truth matching counts."""
    preds, stats = load_airr_tsv(predictions_path)
    scores = {t.case_id: score_case(t, preds.get(t.case_id), reference) for t in truths if t.excluded is None}
    truth_ids = {t.case_id for t in truths}
    matching = {
        "truth_cases": len(truths), "excluded": len(truths) - len(scores), "scored": len(scores),
        "rows": stats["rows"], "predictions": len(preds),
        "duplicates": stats["duplicates"], "invalid_rows": stats["invalid_rows"],
        "missing": sum(cid not in preds for cid in scores),
        "extra": sum(pid not in truth_ids for pid in preds),
        "excluded_ignored": sum(pid in truth_ids and pid not in scores for pid in preds),
    }
    return scores, matching


def _print_matching(tool: str, m: dict) -> None:
    print(f"{tool}: {m['scored']} cases scored ({m['excluded']} excluded of {m['truth_cases']}); "
          f"{m['rows']} rows, {m['duplicates']} duplicates, {m['invalid_rows']} invalid rows; "
          f"missing {m['missing']}, extra {m['extra']}, ignored for excluded cases {m['excluded_ignored']}")


def _slices(scores: dict[str, CaseScores], truths: list[ObservableTruth]) -> dict:
    labels = {t.case_id: slice_labels(t) for t in truths if t.case_id in scores}
    return {dim: by_slice(scores, {cid: lab[dim] for cid, lab in labels.items()}) for dim in SLICE_DIMENSIONS}


def _load_truth(path) -> list[ObservableTruth]:
    truths = read_truth_jsonl(path)
    _check_unique(truths, path)
    return truths


def _check_unique(truths: list[ObservableTruth], source) -> None:
    dup = [cid for cid, n in Counter(t.case_id for t in truths).items() if n > 1]
    if dup:
        raise ValueError(f"{source}: duplicate case ids, e.g. {dup[0]!r} ({len(dup)} in total)")


def _read_jsonl(path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_scores(scores: dict[str, CaseScores], path: Path) -> None:
    """One ``{"case_id", "scores"}`` object per scored case; ``nan`` (applicable, not covered) becomes null."""
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for cid, case in scores.items():
            row = {"case_id": cid, "scores": {k: v if math.isfinite(v) else None for k, v in case.items()}}
            fh.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def _write_report(report: dict, json_path: Path, md_path: Path) -> None:
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")


def _provenance(command: str, inputs: dict[str, str], **extra) -> dict:
    return {
        "command": command,
        "inputs": {k: {"path": str(p), "sha256": _sha256(p)} for k, p in inputs.items()},
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "min_observed": dict(MIN_OBSERVED),
        **extra,
    }


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    """HEAD of the source tree this package runs from, or None outside a git checkout."""
    try:
        r = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    sha = r.stdout.strip() if r.returncode == 0 else ""
    return sha or None
