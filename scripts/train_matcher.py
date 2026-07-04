"""Train AIRRMatch on GenAIRR's multi-locus/species stream; hold out configs for the exam."""
import argparse
import itertools

import torch

from alignair.matcher.config import MatcherConfig
from alignair.matcher.model import AIRRMatch
from alignair.matcher.data import stream_batches
from alignair.gym.curriculum import Curriculum


def _round_robin(configs, cfg, params, batch_size, seed):
    streams = [stream_batches(dc, cfg, params, batch_size, seed + i) for i, dc in enumerate(configs)]
    for batch in itertools.chain.from_iterable(zip(*streams)):
        yield batch


def _to_device(batch, device):
    out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items() if k != "genes"}
    out["genes"] = {}
    for g, gb in batch["genes"].items():
        out["genes"][g] = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in gb.items()}
    return out


def train(configs, cfg, steps, lr, batch_size, out_path, device="cuda",
          progress=0.3, seed=0, log_every=50):
    model = AIRRMatch(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    params = dict(Curriculum().params(progress))
    stream = _round_robin(configs, cfg, params, batch_size, seed)
    model.train()
    for step, batch in zip(range(1, steps + 1), stream):
        batch = _to_device(batch, device)
        loss = model.training_loss(batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            print(f"[{step}/{steps}] loss {loss.item():.4f}", flush=True)
    torch.save({"config": vars(cfg), "model": model.state_dict()}, out_path)
    return model


def _resolve_configs(names):
    import GenAIRR.data as gd
    return [getattr(gd, n) for n in names]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["HUMAN_IGH_OGRDB"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--progress", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=".private/models/airrmatch_v1.pt")
    a = ap.parse_args()
    train(_resolve_configs(a.configs), MatcherConfig(), a.steps, a.lr, a.batch_size,
          a.out, device=a.device, progress=a.progress)
