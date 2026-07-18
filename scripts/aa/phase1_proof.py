"""CLI: run the amino-acid AlignAIR Phase-1 learning proof and freeze the artifact.

Run (GPU): PYTHONPATH=src .venv/bin/python scripts/aa/phase1_proof.py --device cuda
"""
import argparse
import json
import os

from alignair.aa.phase1 import run_learning_proof


def main() -> None:
    ap = argparse.ArgumentParser(description="Amino-acid AlignAIR Phase-1 learning proof")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--heldout-size", type=int, default=4096)
    ap.add_argument("--train-steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--overfit-batch", type=int, default=32)
    ap.add_argument("--overfit-steps", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="experiments/aa/phase1/proof.json")
    ap.add_argument("--manifest", default="experiments/aa/phase1/heldout.jsonl.gz")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    r = run_learning_proof(seeds=tuple(args.seeds), heldout_size=args.heldout_size,
                           train_steps=args.train_steps, batch_size=args.batch_size, lr=args.lr,
                           overfit_batch=args.overfit_batch, overfit_steps=args.overfit_steps,
                           device=args.device, manifest_path=args.manifest)
    with open(args.out, "w") as f:
        json.dump(r, f, indent=2, sort_keys=True)
    # console summary (full class maps + manifest live in the artifact, not the terminal)
    print(f"held-out {r['heldout_size']} reads  fingerprint {r['heldout_fingerprint'][:16]}...")
    for sd, s in sorted(r["seeds"].items()):
        c, b = s["heldout"]["classification"], s["heldout"]["boundaries"]
        o = s["overfit"]
        print(f"  seed {sd} OVERFIT: V {o['v_class_acc']:.3f} J {o['j_class_acc']:.3f}  "
              + " ".join(f"{k}={o[k]:.2f}" for k in
                         ("v_start_mae", "v_end_mae", "j_start_mae", "j_end_mae")))
        for g in ("v", "j"):
            lift = c[g]["paired_lift"]
            ok = lift["lower_gt_zero"] and lift["mean"] >= 0.05
            print(f"  seed {sd} {g.upper()}: acc {c[g]['micro_acc']:.3f} "
                  f"(macro {c[g]['macro_acc']:.3f}, maj {c[g]['majority_baseline']:.3f}) "
                  f"lift {lift['mean']:+.3f} CI[{lift['ci95'][0]:+.3f},{lift['ci95'][1]:+.3f}] "
                  f"{'PASS' if ok else 'FAIL'}")
        for k in ("v_start", "v_end", "j_start", "j_end"):
            imp = b[k]["paired_improvement"]
            ok = imp["lower_gt_zero"] and b[k]["relative_reduction"] >= 0.10
            print(f"  seed {sd} {k}: MAE {b[k]['mae']:.2f} vs const "
                  f"{b[k]['constant_median_baseline_mae']:.2f} "
                  f"({b[k]['relative_reduction']:+.1%}) {'PASS' if ok else 'FAIL'}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
