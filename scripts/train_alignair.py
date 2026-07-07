"""Train the faithful AlignAIR single-chain model on a GenAIRR dataconfig (retrain from scratch)."""
import argparse

import torch

import GenAIRR.data as gd
from alignair.config.alignair_config import AlignAIRConfig
from alignair.models.losses import make_logvars
from alignair.models.single_chain import SingleChainAlignAIR
from alignair.reference.reference_set import ReferenceSet
from alignair.training.alignair_trainer import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataconfig", default="HUMAN_IGH_OGRDB")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--progress", type=float, default=0.5)
    ap.add_argument("--max-len", type=int, default=576)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=".private/models/alignair_single.pt")
    ap.add_argument("--save-every", type=int, default=2000)
    a = ap.parse_args()

    dc = getattr(gd, a.dataconfig)
    ref = ReferenceSet.from_dataconfigs(dc)
    has_d = "D" in ref.genes
    cfg = AlignAIRConfig(max_seq_length=a.max_len, has_d=has_d,
                         v_allele_count=len(ref.gene("V")), j_allele_count=len(ref.gene("J")),
                         d_allele_count=len(ref.gene("D")) if has_d else 0)
    model = SingleChainAlignAIR(cfg)
    logvars = make_logvars(cfg)
    print(f"train {a.dataconfig}: V={cfg.v_allele_count} D={cfg.d_allele_count} J={cfg.j_allele_count} "
          f"has_d={has_d} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    train(model, ref, dc, cfg, logvars, steps=a.steps, batch_size=a.batch_size,
          lr=a.lr, progress=a.progress, device=a.device, save_path=a.out, save_every=a.save_every)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
