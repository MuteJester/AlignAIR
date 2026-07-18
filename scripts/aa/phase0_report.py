"""CLI: build + freeze the amino-acid AlignAIR Phase-0 diagnostic report.

Run: PYTHONPATH=src .venv/bin/python scripts/aa/phase0_report.py --n 200000 --out experiments/aa/phase0
"""
import argparse

from alignair.aa.phase0 import build_phase0_report
from alignair.aa.report import write_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Amino-acid AlignAIR Phase-0 diagnostic report")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--dc", default="HUMAN_IGH_OGRDB")
    ap.add_argument("--all", action="store_true", help="include non-productive reads")
    ap.add_argument("--out", default="experiments/aa/phase0")
    args = ap.parse_args()

    report = build_phase0_report(
        n=args.n, seed=args.seed, rate=args.rate,
        productive_only=not args.all, dc_name=args.dc,
    )
    out = write_report(report, args.out)
    print(f"wrote {out}/report.json and {out}/report.md")


if __name__ == "__main__":
    main()
