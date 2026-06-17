"""DNAlignAIR vs IgBLAST head-to-head on identical GenAIRR strata.

Trains a DNAlignAIR model, then scores BOTH it and IgBLAST against ground truth on
the same records, using the same metrics — so every change is measured against the
incumbent bar. The model is the DEPLOYED end-to-end path (predict_reads).
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from baseline_igblast import gen_records, run_igblast, igblast_to_pred, score, print_scores, GENES  # noqa: E402

import GenAIRR.data as gdata  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.config.dnalignair_config import DNAlignAIRConfig  # noqa: E402
from alignair.core.dnalignair import DNAlignAIR  # noqa: E402
from alignair.losses.dnalignair_loss import DNAlignAIRLoss  # noqa: E402
from alignair.gym.gym import AlignAIRGym  # noqa: E402
from alignair.training.gym_trainer import GymTrainer  # noqa: E402
from alignair.inference.dnalignair_infer import predict_reads  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--aligner", default="softdp")
    ap.add_argument("--region-decoder", default="linear")
    ap.add_argument("--caller", choices=["retrieval", "classifier"], default="retrieval")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    torch.manual_seed(0)
    dc = gdata.HUMAN_IGH_OGRDB
    rs = ReferenceSet.from_dataconfigs(dc)
    counts = {g: len(rs.gene(g).names) for g in ("V", "D", "J")} if rs.has_d \
        else {g: len(rs.gene(g).names) for g in ("V", "J")}
    cfg = DNAlignAIRConfig(d_model=args.d_model, n_layers=args.layers,
                           aligner=args.aligner, region_decoder=args.region_decoder,
                           caller=args.caller, allele_counts=counts)
    model = DNAlignAIR(cfg)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d, use_boundary=(args.region_decoder == "query"))
    gym = AlignAIRGym([dc], rs, seed=0)
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=5e-4, batch_size=args.batch)
    print(f"training {sum(p.numel() for p in model.parameters())/1e6:.2f}M params "
          f"for {args.steps} steps (aligner={args.aligner}, region={args.region_decoder})...")
    trainer.fit(total_steps=args.steps, global_total=args.steps)

    strata = [("clean", 0.0, None), ("moderate", 0.5, None),
              ("hard", 1.0, None), ("fragment~80bp", 1.0, 80)]
    print(f"\n=== DNAlignAIR vs IgBLAST | n={args.n}/stratum ===")
    for name, p, crop in strata:
        recs = gen_records(p, args.n, args.seed, crop)
        reads = [r["sequence"] for r in recs]
        model_preds = predict_reads(model, rs, reads)
        igb_preds = [igblast_to_pred(r) for r in run_igblast(recs)]
        print(f"\n[{name}]")
        print(" IgBLAST:")
        print_scores(score(recs, igb_preds), indent="   ")
        print(" DNAlignAIR:")
        print_scores(score(recs, model_preds), indent="   ")


if __name__ == "__main__":
    main()
