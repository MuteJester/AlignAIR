"""Train XAttnAligner (LLM-encoder aligner) on the stratified GenAIRR gym with the four-task loss
(orientation + region + retrieval InfoNCE + allele set-NCE + germline-span), AdamW + cosine + bf16.

Memory-fit: the matcher is chunked over candidates (cand_chunk) and the reference is re-encoded under
no_grad every `refresh` steps (the shared encoder still learns germline structure via the read path),
so it fits a 24GB GPU at full IGH scale.

Monitoring: appends a JSON line per logged step to --log (loss parts, lr, reads/s, eval), and writes
`<save>` + `<save>.latest` checkpoints. Tail the log or read the JSONL to watch progress live.

24h run (background):
  PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./.venv/bin/python \
      scripts/train_xattn.py --locus igh --d-model 128 --n-layers 8 --batch-size 24 \
      --hours 24 --steps 2000000 --cand-chunk 4 --refresh 25 \
      --save .private/models/xattn_igh.pt --log .private/models/xattn_igh.jsonl
"""
import argparse
import json
import math
import os
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
def _eval(model, rs, dc, ref_emb, device, n, bs, seed_m, cand_chunk):
    gym = AlignAIRGym([dc], rs, n=n, seed=123)
    loader = DataLoader(gym, batch_size=bs, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))
    hit = tot = cstart = 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch["tokens"], batch["mask"], ref_emb, seed_m=seed_m, reference_set=rs,
                    cand_chunk=cand_chunk)
        best = out["match"]["V"]["best_global_idx"]
        mh = batch["v_allele"]
        hit += int((mh.gather(1, best[:, None]).squeeze(1) > 0).sum())
        cstart += int((out["match"]["V"]["germ_start"] == batch["v_germline_start"]).sum())
        tot += best.shape[0]
    return {"v_allele_acc": hit / max(tot, 1), "v_gstart_exact": cstart / max(tot, 1)}


def _log(path, rec):
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


def train_xattn(cfg, dc, steps, batch_size, lr, device, save=None, ckpt_every=2000,
                eval_n=200, seed_m=4, warmup=200, grad_clip=1.0, progress=True,
                hours=None, refresh=25, cand_chunk=4, n_sib=6, n_rand=6, log=None, resume=None):
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(dc)
    model = XAttnAligner(cfg).to(device).train()
    sib = build_sibling_index(rs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    start = 0
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); start = int(ck.get("step", 0))
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        print(f"resumed from {resume} at step {start}", flush=True)
    gym = AlignAIRGym([dc], rs, n=batch_size * 8, seed=0, curriculum=StratifiedCurriculum())
    loader = DataLoader(gym, batch_size=batch_size, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    rng = random.Random(0)
    it = iter(loader)
    t0 = time.perf_counter()
    ref_emb = None
    step = start
    deadline = (t0 + hours * 3600) if hours else None
    while step < steps and (deadline is None or time.perf_counter() < deadline):
        if ref_emb is None or step % refresh == 0:
            with torch.no_grad():                                   # detached cache -> bounds memory
                ref_emb = model.encode_reference(rs)
        for pg in opt.param_groups:
            pg["lr"] = _cosine_lr(step, steps, warmup, lr)
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            total, parts = xattn_losses(model, batch, ref_emb, sib, rng,
                                        n_sib=n_sib, n_rand=n_rand, cand_chunk=cand_chunk)
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if progress and (step % 50 == 0):
            el = time.perf_counter() - t0
            rps = (step - start + 1) * batch_size / max(el, 1e-9)
            rec = {"step": step, "t": round(el, 1), "loss": round(float(total), 4),
                   "lr": round(opt.param_groups[0]["lr"], 6), "reads_s": round(rps, 1),
                   **{k: round(float(v), 4) for k, v in parts.items()}}
            print(f"[{step:7d}/{steps}] loss {rec['loss']:.3f} "
                  + " ".join(f"{k}={rec[k]}" for k in parts) + f"  {rec['reads_s']:.0f} r/s", flush=True)
            _log(log, rec)
        if save and ((step + 1) % ckpt_every == 0):
            blob = {"config": cfg.to_dict(), "model": model.state_dict(), "step": step + 1,
                    "opt": opt.state_dict()}
            torch.save(blob, save); torch.save(blob, save + ".latest")
        step += 1
    if save:
        torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": step,
                    "opt": opt.state_dict()}, save)
    if eval_n:
        model.eval()
        with torch.no_grad():
            ev = _eval(model, rs, dc, model.encode_reference(rs), device, eval_n, batch_size,
                       seed_m, cand_chunk)
        print(f"\neval (n={eval_n}): {ev}", flush=True)
        _log(log, {"step": step, "eval": ev})
        model.train()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--hours", type=float, default=None, help="wall-clock stop (e.g. 24)")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--locus", default="igh", choices=["igh", "igk"])
    ap.add_argument("--cand-chunk", type=int, default=4, help="matcher candidate chunk (memory)")
    ap.add_argument("--refresh", type=int, default=25, help="re-encode reference every N steps")
    ap.add_argument("--eval-n", type=int, default=250)
    ap.add_argument("--seed-m", type=int, default=4)
    ap.add_argument("--save", default="")
    ap.add_argument("--log", default="")
    ap.add_argument("--resume", default="")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dc = {"igh": gdata.HUMAN_IGH_OGRDB, "igk": gdata.HUMAN_IGK_OGRDB}[a.locus]
    cfg = DNAlignAIRConfig(d_model=a.d_model, n_layers=a.n_layers, nhead=a.nhead,
                           dim_feedforward=2 * a.d_model, backbone="shared")
    if a.save:
        os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
    train_xattn(cfg, dc, a.steps, a.batch_size, a.lr, device, save=a.save or None,
                ckpt_every=a.ckpt_every, eval_n=a.eval_n, seed_m=a.seed_m, hours=a.hours,
                refresh=a.refresh, cand_chunk=a.cand_chunk, log=a.log or None,
                resume=a.resume or None)


if __name__ == "__main__":
    main()
