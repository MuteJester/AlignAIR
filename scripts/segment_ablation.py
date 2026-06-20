"""Oracle-vs-predicted segment ablation (codex gate for the CRF / soft-boundary work).

The reader/matcher is fed the segment selected by the region head's argmax. If feeding
the TRUE (oracle) region labels instead lifts call accuracy a lot, the hard segmentation
interface is a real bottleneck and the monotone-CRF + boundary-marginal soft-DP is worth
building. If oracle ~= predicted, segmentation is NOT what caps (e.g.) heavy-SHM V — look
elsewhere. We report per-gene top-1-in-set call accuracy under both, across difficulties.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

import GenAIRR.data as gdata  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.config.dnalignair_config import DNAlignAIRConfig  # noqa: E402
from alignair.core.dnalignair import DNAlignAIR  # noqa: E402
from alignair.gym.gym import AlignAIRGym  # noqa: E402
from alignair.gym.curriculum import Curriculum  # noqa: E402
from alignair.training.gym_trainer import GymTrainer  # noqa: E402


def call_acc(match_G, allele_multihot):
    """top-1-in-set accuracy: argmax allele is in the example's true allele set."""
    pred = match_G.argmax(dim=-1)                              # (B,)
    hit = allele_multihot.gather(1, pred.unsqueeze(1)).squeeze(1)  # (B,) 1 if in set
    return float(hit.float().mean())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_ramp.pt")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    ckpt = torch.load(args.model, map_location=device)
    cfg = DNAlignAIRConfig(**ckpt["config"]) if isinstance(ckpt["config"], dict) else ckpt["config"]
    model = DNAlignAIR(cfg).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    print(f"loaded {args.model}  ({sum(p.numel() for p in model.parameters())/1e6:.1f}M)")

    gym = AlignAIRGym([gdata.HUMAN_IGH_OGRDB], rs, seed=0, curriculum=Curriculum())
    trainer = GymTrainer(model, torch.nn.Identity(), rs, gym, batch_size=args.batch, device=device)
    genes = ["v", "j"] + (["d"] if rs.has_d else [])
    ref_emb = model.encode_reference(rs)

    for label, p in [("clean", 0.0), ("moderate", 0.5), ("hard/heavy-SHM", 1.0)]:
        gym.set_progress(p)
        loader = trainer._loader()
        ora = {g: [] for g in genes}
        pre = {g: [] for g in genes}
        for nb, batch in enumerate(loader):
            if nb >= args.batches:
                break
            batch = trainer._to_device(batch)
            out = model(batch["tokens"], batch["mask"], ref_emb,
                        orientation_ids=batch["orientation_id"])
            pred_region = out["region_logits"].argmax(-1)
            true_region = batch["region_labels"]
            m_oracle = model.match_alleles(out["canon_tokens"], batch["mask"], true_region,
                                           ref_emb, reps=out["reps"])
            m_pred = model.match_alleles(out["canon_tokens"], batch["mask"], pred_region,
                                         ref_emb, reps=out["reps"])
            for g in genes:
                G = g.upper()
                ora[g].append(call_acc(m_oracle[G], batch[f"{g}_allele"]))
                pre[g].append(call_acc(m_pred[G], batch[f"{g}_allele"]))
        print(f"\n[{label}]")
        for g in genes:
            o = sum(ora[g]) / len(ora[g]); q = sum(pre[g]) / len(pre[g])
            print(f"  {g.upper()}: oracle-seg={o:.3f}  predicted-seg={q:.3f}  gap={o - q:+.3f}")


if __name__ == "__main__":
    main()
