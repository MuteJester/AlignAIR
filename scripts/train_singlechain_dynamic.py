"""Train SingleChainAlignAIR in DYNAMIC-reference mode on the GenAIRR gym.

The read's V/D/J segment is matched (cosine + temperature, InfoNCE) against the reference germlines
encoded through the SAME conv encoder — so identity comes from sequence relationship, not a memorized
head. To FORCE (and measure) generalization to unseen alleles we hold out a fraction of alleles per
gene: they are masked out of the candidate set during training, and evaluated separately.

  train candidate set = train alleles (minus a per-batch random subset of negatives)
  held-out eval       = accuracy on reads whose true allele was NEVER trained on

Run:
  PYTHONPATH=src ./.venv/bin/python scripts/train_singlechain_dynamic.py \
      --steps 20000 --batch-size 64 --heldout-frac 0.2 --save .private/models/singlechain_dyn.pt
"""
import argparse
import time

import torch
from torch.utils.data import DataLoader
import GenAIRR.data as gdata

from alignair.reference.reference_set import ReferenceSet
from alignair.gym import AlignAIRGym, gym_collate
from alignair.gym.curriculum import Curriculum
from alignair.nn.singlechain import SingleChainAlignAIR, hierarchical_loss
from alignair.nn.singlechain.data import singlechain_inputs
from alignair.nn.singlechain.dynamic import AlleleBank

GENES = ["V", "J", "D"]


def _to_device(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def evaluate(model, collated, bank, heldout, max_len, device):
    """Full reference active; report calling accuracy split by train vs held-out true allele."""
    model.eval()
    tokens, _ = singlechain_inputs(collated, max_len, True, device)
    out = model(tokens, bank.tokens)
    rep = {}
    for G in GENES:
        g = G.lower()
        mh = collated[f"{g}_allele"].to(device)
        prim = collated[f"{g}_primary_idx"].to(device)
        present = (mh.sum(-1) > 0) & (prim >= 0)
        pred = out[f"{g}_allele_scores"].argmax(-1)
        hit = mh.gather(1, pred.unsqueeze(1)).squeeze(1) > 0
        is_held = torch.zeros_like(present)
        is_held[present] = heldout[G].to(device)[prim[present]]
        for name, sel in (("train", present & ~is_held), ("held", present & is_held)):
            n = int(sel.sum())
            rep[f"{G}/{name}"] = (float((hit & sel).sum()) / n) if n else float("nan")
    model.train()
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seq-length", type=int, default=576)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--heldout-frac", type=float, default=0.2)
    ap.add_argument("--subset-keep", type=float, default=0.5, help="per-batch fraction of train negatives kept")
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--save", default="")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    dc = gdata.HUMAN_IGH_OGRDB
    rs = ReferenceSet.from_dataconfigs(dc)
    counts = {G: len(rs.gene(G).names) for G in GENES}
    bank = AlleleBank(rs, a.max_seq_length).to(device)

    # deterministic held-out allele split per gene (every 1/frac-th allele)
    heldout, train_mask = {}, {}
    step = max(int(round(1 / a.heldout_frac)), 2)
    for G in GENES:
        h = torch.zeros(counts[G], dtype=torch.bool)
        h[::step] = True
        heldout[G] = h.to(device)
        train_mask[G] = (~h).to(device)
    print(f"[dynamic] reference={counts}  held-out per gene="
          f"{ {G:int(heldout[G].sum()) for G in GENES} }", flush=True)

    model = SingleChainAlignAIR(a.max_seq_length, counts["V"], counts["J"], counts["D"],
                                allele_mode="dynamic").to(device)
    model(torch.zeros(2, a.max_seq_length, dtype=torch.long, device=device), bank.tokens)  # init lazy
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    print(f"[dynamic] params={sum(p.numel() for p in model.parameters())/1e6:.1f}M device={device}", flush=True)

    gym = AlignAIRGym([dc], rs, n=a.batch_size * 8, seed=0, curriculum=Curriculum())
    loader = DataLoader(gym, batch_size=a.batch_size, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))

    eval_batch, done, t0 = None, 0, time.perf_counter()
    model.train()
    while done < a.steps:
        for batch in loader:
            if done >= a.steps:
                break
            batch = _to_device(batch, device)
            if eval_batch is None:
                eval_batch = batch
            tokens, tgt = singlechain_inputs(batch, a.max_seq_length, True, device)

            # per-batch active candidate set: train alleles, keep all batch positives + random negatives
            cmask = {}
            for G in GENES:
                pos = batch[f"{G.lower()}_allele"].sum(0) > 0
                keep = (torch.rand(counts[G], generator=gen).to(device) < a.subset_keep) | pos
                active = train_mask[G] & keep
                cmask[G] = active
                tgt[f"{G.lower()}_allele"] = tgt[f"{G.lower()}_allele"] * active.float()  # drop inactive positives

            out = model(tokens, bank.tokens, candidate_mask=cmask)
            loss, logs = hierarchical_loss(model, out, tgt)
            if not torch.isfinite(loss):
                done += 1
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()
            gym.set_progress(min(1.0, done / max(a.steps, 1)))
            done += 1

            if done % a.log_every == 0:
                el = time.perf_counter() - t0
                print(f"[{done}/{a.steps}] loss={logs['total']:.3f} seg={logs['segmentation']:.3f} "
                      f"clf={logs['classification']:.3f}  {el/done:.2f}s/step", flush=True)
            if done % a.eval_every == 0 and eval_batch is not None:
                rep = evaluate(model, eval_batch, bank, heldout, a.max_seq_length, device)
                s = "  ".join(f"{k}={v:.3f}" for k, v in rep.items())
                print(f"    eval: {s}", flush=True)
            if a.save and done % a.ckpt_every == 0:
                torch.save({"model": model.state_dict(), "args": vars(a), "counts": counts}, a.save)

    if a.save:
        torch.save({"model": model.state_dict(), "args": vars(a), "counts": counts}, a.save)
        print(f"[done] saved -> {a.save}", flush=True)


if __name__ == "__main__":
    main()
