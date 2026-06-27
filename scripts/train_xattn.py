"""Train XAttnAligner (LLM-encoder aligner) on the stratified GenAIRR gym with the four-task loss
(orientation + region + allele set-NCE + germline-span), AdamW + cosine schedule + bf16, with
checkpoint/resume and a set-aware allele-accuracy readout.

Run:
  PYTHONPATH=src ./.venv/bin/python scripts/train_xattn.py --d-model 128 --n-layers 8 \
      --steps 20000 --batch-size 64 --locus igh --save .private/models/xattn_d128.pt
"""
import argparse
import math
import random
import time

import torch
import GenAIRR.data as gdata
from torch.utils.data import DataLoader

from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.xattn_aligner import XAttnAligner
from alignair.reference.reference_set import ReferenceSet
from alignair.training.reader import build_sibling_index
from alignair.training.xattn_loss import xattn_losses
from alignair.gym import AlignAIRGym, gym_collate
from alignair.gym.curriculum import StratifiedCurriculum


def _cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base * (1 + math.cos(math.pi * min(p, 1.0)))


@torch.no_grad()
def _allele_accuracy(model, rs, dc, ref_emb, device, n, bs, seed_m):
    gym = AlignAIRGym([dc], rs, n=n, seed=123)
    loader = DataLoader(gym, batch_size=bs, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))
    hit = tot = 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch["tokens"], batch["mask"], ref_emb, seed_m=seed_m, reference_set=rs)
        best = out["match"]["V"]["best_global_idx"]                      # (B,)
        mh = batch["v_allele"]                                          # (B,K) set
        hit += int((mh.gather(1, best[:, None]).squeeze(1) > 0).sum()); tot += best.shape[0]
    return hit / max(tot, 1)


def train_xattn(cfg, dc, steps, batch_size, lr, device, save=None, ckpt_every=2000,
                eval_n=200, seed_m=4, warmup=200, grad_clip=1.0, progress=True):
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(dc)
    model = XAttnAligner(cfg).to(device).train()
    sib = build_sibling_index(rs)
    gym = AlignAIRGym([dc], rs, n=batch_size * 8, seed=0, curriculum=StratifiedCurriculum())
    loader = DataLoader(gym, batch_size=batch_size, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    rng = random.Random(0)
    it = iter(loader)
    t0 = time.perf_counter()
    for step in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = _cosine_lr(step, steps, warmup, lr)
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        ref_emb = model.encode_reference(rs)                            # re-encode (weights move)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            total, parts = xattn_losses(model, batch, ref_emb, sib, rng)
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if progress and (step % 100 == 0 or step == steps - 1):
            el = time.perf_counter() - t0
            print(f"[{step:6d}/{steps}] loss {float(total):.3f} "
                  + " ".join(f"{k}={float(v):.3f}" for k, v in parts.items())
                  + f"  {el:.0f}s", flush=True)
        if save and ((step + 1) % ckpt_every == 0 or step == steps - 1):
            torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": step + 1}, save)
    if eval_n:
        model.eval()
        acc = _allele_accuracy(model, rs, dc, model.encode_reference(rs), device, eval_n,
                               batch_size, seed_m)
        print(f"\nset-aware V allele accuracy (n={eval_n}, seed_m={seed_m}): {acc:.3f}", flush=True)
        model.train()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--locus", default="igh", choices=["igh", "igk"])
    ap.add_argument("--eval-n", type=int, default=250)
    ap.add_argument("--seed-m", type=int, default=4)
    ap.add_argument("--save", default="")
    ap.add_argument("--ckpt-every", type=int, default=2000)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dc = {"igh": gdata.HUMAN_IGH_OGRDB, "igk": gdata.HUMAN_IGK_OGRDB}[a.locus]
    cfg = DNAlignAIRConfig(d_model=a.d_model, n_layers=a.n_layers, nhead=a.nhead,
                           dim_feedforward=2 * a.d_model, backbone="shared")
    train_xattn(cfg, dc, a.steps, a.batch_size, a.lr, device, save=a.save or None,
                ckpt_every=a.ckpt_every, eval_n=a.eval_n, seed_m=a.seed_m)


if __name__ == "__main__":
    main()
