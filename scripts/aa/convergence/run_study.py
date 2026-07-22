"""Convergence-study CLI: continue training the frozen Stage-1 incumbent for a declared budget,
checkpoint on a fixed schedule, and evaluate every checkpoint (incumbent + candidates) on the
same frozen §3 cohort. See docs/superpowers/specs/2026-07-22-aa-convergence-study-design.md §4/§8.

This is orchestration only -- it does NOT run §5 selection (a separate, later, auditable step
against the ``checkpoints_metrics.json`` this produces) and does NOT touch the frozen incumbent
(``--resume``) or any other frozen snapshot; continuation checkpoints are written under
``experiments/aa/strategic/`` (gitignored) as immutable ``model_cont_step{N}.pt`` files.

Run (the real study; NOT run as part of this task):
  PYTHONPATH=src .venv/bin/python scripts/aa/convergence/run_study.py \\
      --resume experiments/aa/strategic/model_stage1_PASS.pt --budget 300000 \\
      --ckpt-steps 25000,75000,150000,300000 --device cuda \\
      --art-dir docs/superpowers/artifacts/2026-07-22-aa-convergence
"""
from __future__ import annotations

import argparse

from alignair.aa.convergence.driver import run_study

DEFAULT_RESUME = "experiments/aa/strategic/model_stage1_PASS.pt"
DEFAULT_ART_DIR = "docs/superpowers/artifacts/2026-07-22-aa-convergence"


def _parse_ckpt_steps(s: str) -> tuple:
    return tuple(int(x) for x in s.split(",") if x.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume", default=DEFAULT_RESUME,
                    help="incumbent checkpoint to continue from (never modified)")
    ap.add_argument("--budget", type=int, default=300000, help="+N continuation steps")
    ap.add_argument("--ckpt-steps", type=_parse_ckpt_steps, default="25000,75000,150000,300000",
                    help="comma-separated RELATIVE checkpoint offsets from the incumbent's step "
                         "(e.g. '25000,75000,150000,300000' -> +25k/+75k/+150k/+300k)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--art-dir", default=DEFAULT_ART_DIR,
                    help="artifact output directory (§8): cohort_fingerprint.json, "
                         "checkpoints_metrics.json, convergence.md, predeclaration.md")
    ap.add_argument("--n-per-stratum", type=int, default=200,
                    help="frozen evaluator cohort size per stratum")
    ap.add_argument("--cohort-seed", type=int, default=20250722,
                    help="frozen evaluator cohort seed (disjoint from training seed 0 and "
                         "calibrate_stage2's 1000/9000)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--val-every", type=int, default=25000)
    ap.add_argument("--plateau-k", type=int, default=3,
                    help="log (informational only) when the validation objective has not "
                         "improved over this many consecutive validations")
    args = ap.parse_args()

    run_study(args.resume, args.budget, args.ckpt_steps, args.device, args.art_dir,
              cohort_seed=args.cohort_seed, n_per_stratum=args.n_per_stratum,
              batch_size=args.batch_size, val_every=args.val_every, plateau_k=args.plateau_k)


if __name__ == "__main__":
    main()
