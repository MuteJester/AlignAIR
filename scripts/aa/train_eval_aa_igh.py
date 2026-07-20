"""Train the amino-acid AlignAIR with the SAME scheme as the shipped nucleotide IGH model
(only the input is AA and coordinates are AA-space), then evaluate whether allele-call ambiguity
emerges on its own.

Run (GPU): PYTHONPATH=src .venv/bin/python scripts/aa/train_eval_aa_igh.py --device cuda --steps 20000
"""
import argparse
import itertools
import json
import os

import torch

from alignair.aa.data import build_class_maps
from alignair.aa.eval_gym import evaluate_aa
from alignair.aa.gym_bridge import aa_usable
from alignair.aa.train_gym import setup_aa, train_aa, _usable_stream


def _heldout(dc_name, n, seed, max_seq_length):
    stream = _usable_stream(dc_name, (0.3, 0.6, 0.9), 0.25, seed, 1, None)
    return list(itertools.islice(stream, n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dc", default="HUMAN_IGH_OGRDB")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--heldout-size", type=int, default=4096)
    ap.add_argument("--out", default="experiments/aa/same_scheme")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    model, ref, cfg, logvars = setup_aa(args.dc, max_seq_length=192)
    print(f"AA model: {sum(p.numel() for p in model.parameters()):,} params | "
          f"V/D/J alleles {cfg.v_allele_count}/{cfg.d_allele_count}/{cfg.j_allele_count} | "
          f"orientation_head={model.orientation_head}", flush=True)

    history = train_aa(model, ref, args.dc, cfg, logvars, steps=args.steps,
                       batch_size=args.batch_size, lr=args.lr, seed=args.seed, device=args.device,
                       log_every=250, grad_clip=1.0, save_path=os.path.join(args.out, "model.pt"),
                       save_every=5000)

    # protein-equivalence class maps (name -> class id) for the emergent-ambiguity metric
    cmaps, meta = build_class_maps(args.dc)
    eval_class_maps = {"V": cmaps["V"].allele_class, "J": cmaps["J"].allele_class}

    heldout = _heldout(args.dc, args.heldout_size, 987_654_321, 192)
    metrics = evaluate_aa(model, heldout, ref, cfg, eval_class_maps, device=args.device)
    metrics["train_steps"] = args.steps
    metrics["final_train_loss"] = history[-1]["total"] if history else None
    metrics["n_v_classes"] = meta["n_v_classes"]
    metrics["n_j_classes"] = meta["n_j_classes"]

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print("\n==== EMERGENT-AMBIGUITY EVALUATION ====", flush=True)
    print(f"held-out {metrics['n']} reads | final train loss {metrics['final_train_loss']:.3f}")
    for g in ("v", "d", "j"):
        a = metrics["alleles"][g]
        line = f"  {g.upper()}: nt_top1 {a['nt_top1']:.3f}  true_in_set {a['true_in_predicted_set']:.3f}  set_size {a['mean_pred_set_size']:.2f}"
        if "class_top1" in a:
            line += (f"  |  class_top1 {a['class_top1']:.3f}  "
                     f"nt_errors_within_class {a['frac_errors_within_class']:.3f}")
        print(line, flush=True)
    print(f"  seg MAE (residues): " + " ".join(
        f"{k}={v:.2f}" for k, v in metrics["segmentation_mae"].items()))
    print(f"  mutation_mae {metrics['mutation_mae']:.4f}  productive_acc {metrics['productive_acc']:.3f}")
    print(f"\nwrote {args.out}/metrics.json + model.pt", flush=True)


if __name__ == "__main__":
    main()
