"""Train the PyTorch SingleChainAlignAIR (conv, segmentation-first) on the online GenAIRR gym.

Fixed-reference AlignAIR port. Reports per-gene allele-calling accuracy (set-aware) and boundary
MAE (nt) so we can compare against the transformer detector and AlignAIR's published numbers.

Run:
  PYTHONPATH=src ./.venv/bin/python scripts/train_singlechain.py \
      --steps 20000 --batch-size 64 --save .private/models/singlechain_igh.pt
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


def _to_device(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def evaluate(model, collated, max_len, has_d, device):
    model.eval()
    tokens, tgt = singlechain_inputs(collated, max_len, has_d, device)
    out = model(tokens)
    rep = {}
    for g in (["v", "j"] + (["d"] if has_d else [])):
        mh = collated[f"{g}_allele"].to(device)
        present = mh.sum(-1) > 0
        pred = out[f"{g}_allele"].argmax(-1)
        hit = (mh.gather(1, pred.unsqueeze(1)).squeeze(1) > 0) & present   # pred is in the true set
        acc = float(hit.sum()) / max(int(present.sum()), 1)
        se = (out[f"{g}_start"].squeeze(-1) - tgt[f"{g}_start"]).abs()[present]
        ee = (out[f"{g}_end"].squeeze(-1) - tgt[f"{g}_end"]).abs()[present]
        rep[g] = (acc, float(se.mean()) if present.any() else float("nan"),
                  float(ee.mean()) if present.any() else float("nan"))
    model.train()
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seq-length", type=int, default=576)
    ap.add_argument("--filter-size", type=int, default=128)
    ap.add_argument("--feature-dim", type=int, default=576)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--locus", default="igh", choices=["igh", "igk"])
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--save", default="")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    dc = {"igh": gdata.HUMAN_IGH_OGRDB, "igk": gdata.HUMAN_IGK_OGRDB}[a.locus]
    rs = ReferenceSet.from_dataconfigs(dc)
    has_d = rs.has_d
    counts = {G: len(rs.gene(G).names) for G in (["V", "J"] + (["D"] if has_d else []))}
    model = SingleChainAlignAIR(a.max_seq_length, counts["V"], counts["J"],
                                counts.get("D"), filter_size=a.filter_size,
                                feature_dim=a.feature_dim).to(device)
    # init lazy layers
    model(torch.zeros(2, a.max_seq_length, dtype=torch.long, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    gym = AlignAIRGym([dc], rs, n=a.batch_size * 8, seed=0, curriculum=Curriculum())
    loader = DataLoader(gym, batch_size=a.batch_size,
                        collate_fn=lambda b: gym_collate(b, rs, rs.has_d))
    print(f"[singlechain] locus={a.locus} reference={counts} params="
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M device={device}", flush=True)

    eval_batch, done, t0 = None, 0, time.perf_counter()
    model.train()
    while done < a.steps:
        for batch in loader:
            if done >= a.steps:
                break
            batch = _to_device(batch, device)
            if eval_batch is None:
                eval_batch = batch
            tokens, tgt = singlechain_inputs(batch, a.max_seq_length, has_d, device)
            out = model(tokens)
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
                rep = evaluate(model, eval_batch, a.max_seq_length, has_d, device)
                s = "  ".join(f"{g.upper()} acc={r[0]:.3f} startMAE={r[1]:.1f} endMAE={r[2]:.1f}"
                              for g, r in rep.items())
                print(f"    eval: {s}", flush=True)
            if a.save and done % a.ckpt_every == 0:
                torch.save({"model": model.state_dict(), "args": vars(a), "counts": counts}, a.save)

    if a.save:
        torch.save({"model": model.state_dict(), "args": vars(a), "counts": counts}, a.save)
        print(f"[done] saved -> {a.save}", flush=True)


if __name__ == "__main__":
    main()
