"""Train AIRRMatch on GenAIRR's multi-locus/species stream with per-gene convergence telemetry.

Logs (console + JSONL) the signals that matter for this model's health:
  - per-gene selection loss, top-1-in-set accuracy, and true-vs-best-sibling MARGIN
    (D is the canary; the margin is the SNP-discrimination health signal),
  - periodic HELD-OUT accuracy vs IN-DISTRIBUTION accuracy and the generalization GAP
    (the spec's A->B/C decision gate),
  - grad-norm and reads/sec.
"""
import argparse
import itertools
import json
import time

import torch

from alignair.matcher.config import MatcherConfig
from alignair.matcher.model import AIRRMatch
from alignair.matcher.data import stream_batches
from alignair.matcher.telemetry import batch_metrics
from alignair.matcher.evaluate import evaluate
from alignair.gym.curriculum import Curriculum


def _resolve(name):
    import GenAIRR.data as gd
    return getattr(gd, name)


def _round_robin(named_configs, cfg, params, batch_size, seed):
    streams = [stream_batches(dc, cfg, params, batch_size, seed + i)
               for i, (_, dc) in enumerate(named_configs)]
    for batch in itertools.chain.from_iterable(zip(*streams)):
        yield batch


def _to_device(batch, device):
    out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items() if k != "genes"}
    out["genes"] = {}
    for g, gb in batch["genes"].items():
        out["genes"][g] = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in gb.items()}
    return out


def _grad_norm(model):
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().norm()) ** 2
    return sq ** 0.5


def _fmt_acc(d):
    return " ".join(f"{g}={d[g]:.2f}" for g in d)


def train(train_names, heldout_names, cfg, steps, lr, batch_size, out_path,
          device="cuda", progress=0.3, seed=0, log_every=50, eval_every=2000,
          eval_n=200, save_every=5000, log_path=None):
    train_cfgs = [(n, _resolve(n)) for n in train_names]
    heldout_cfgs = [(n, _resolve(n)) for n in heldout_names]
    id_name, id_cfg = train_cfgs[0]

    model = AIRRMatch(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    params = dict(Curriculum().params(progress))
    stream = _round_robin(train_cfgs, cfg, params, batch_size, seed)

    logf = open(log_path, "a") if log_path else None

    def emit(rec):
        if logf:
            logf.write(json.dumps(rec) + "\n")
            logf.flush()

    print(f"train={train_names} heldout={heldout_names} steps={steps} bs={batch_size} "
          f"d_model={cfg.d_model} layers={cfg.n_layers}", flush=True)
    model.train()
    t0 = time.time()
    seen = 0
    for step, batch in zip(range(1, steps + 1), stream):
        batch = _to_device(batch, device)
        gs = model._gene_scores(batch)
        loss = model._reduce_loss(gs)
        opt.zero_grad()
        loss.backward()
        gn = _grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += batch["read_ids"].shape[0]
        lv = float(loss.detach())

        if step % log_every == 0:
            gm = batch_metrics({g: [(s.detach(), t) for s, t in recs] for g, recs in gs.items()},
                               cfg.tau, cfg.margin)
            rps = seen / (time.time() - t0)
            emit({"t": "train", "step": step, "loss": lv, "grad_norm": gn,
                  "reads_per_s": rps,
                  "gene": {g: {k: gm[g][k] for k in ("loss", "acc", "margin", "n")} for g in gm}})
            genes = " | ".join(
                f"{g} L={gm[g]['loss']:.2f} acc={gm[g]['acc']:.2f} mgn={gm[g]['margin']:+.2f}"
                for g in gm)
            print(f"[{step}/{steps}] loss {lv:.3f} gnorm {gn:.1f} {rps:.0f}rd/s | {genes}",
                  flush=True)

        if eval_every and step % eval_every == 0:
            model.eval()
            id_acc = evaluate(model, id_cfg, cfg, n=eval_n, progress=progress, seed=99, device=device)
            ho = {name: evaluate(model, dc, cfg, n=eval_n, progress=progress, seed=99, device=device)
                  for name, dc in heldout_cfgs}
            model.train()
            emit({"t": "eval", "step": step, "in_dist": {id_name: id_acc}, "held_out": ho})
            print(f"  [eval @ {step}] IN-DIST {id_name}: {_fmt_acc(id_acc)}", flush=True)
            for name, acc in ho.items():
                gap = {g: id_acc.get(g, 0.0) - acc.get(g, 0.0) for g in acc}
                print(f"             HELD-OUT {name}: {_fmt_acc(acc)}  |  gap {_fmt_acc(gap)}",
                      flush=True)

        if save_every and step % save_every == 0:
            torch.save({"config": vars(cfg), "model": model.state_dict(), "step": step}, out_path)

    torch.save({"config": vars(cfg), "model": model.state_dict(), "step": steps}, out_path)
    if logf:
        logf.close()
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=["HUMAN_IGH_OGRDB", "HUMAN_IGK_OGRDB",
                                                   "HUMAN_IGL_OGRDB", "MOUSE_IGH_IMGT"])
    ap.add_argument("--heldout", nargs="+", default=["RHESUS_IGH_IMGT"])
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--progress", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--out", default=".private/models/airrmatch_v1.pt")
    ap.add_argument("--log", default=".private/logs/airrmatch_v1.jsonl")
    a = ap.parse_args()
    train(a.train, a.heldout, MatcherConfig(), a.steps, a.lr, a.batch_size, a.out,
          device=a.device, progress=a.progress, log_every=a.log_every,
          eval_every=a.eval_every, eval_n=a.eval_n, log_path=a.log)
