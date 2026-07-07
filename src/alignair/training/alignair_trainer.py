"""Training loop for the faithful AlignAIR model on the GenAIRR gym stream.

``build_batch`` adapts the reusable ``gym_collate`` output (tokens, query coords, allele multi-hot,
mutation/indel/productive) into the model's input + target dicts. ``train_step`` runs one
optimizer step and applies the TF weight *constraints* (analysis-head kernel clamps + Kendall
log-var clamps) after ``optimizer.step()``.
"""
from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F

from ..config.alignair_config import AlignAIRConfig
from ..gym import build_experiment, build_targets, gym_collate


def build_batch(records, reference_set, cfg: AlignAIRConfig, device: str = "cpu"):
    """(records, reference) -> (model_input, targets), tokens padded/truncated to max_seq_length."""
    bundles = [build_targets(r, reference_set, cfg.has_d) for r in records]
    b = gym_collate(bundles, reference_set, cfg.has_d)
    L = cfg.max_seq_length
    tok = b["tokens"]
    tok = F.pad(tok, (0, L - tok.shape[1])) if tok.shape[1] < L else tok[:, :L]
    batch_in = {"tokenized_sequence": tok.to(device)}

    genes = ["v", "j"] + (["d"] if cfg.has_d else [])
    targets: dict = {}
    for g in genes:
        targets[f"{g}_start"] = b[f"{g}_start"].float().unsqueeze(-1).to(device)
        targets[f"{g}_end"] = b[f"{g}_end"].float().unsqueeze(-1).to(device)
        targets[f"{g}_allele"] = b[f"{g}_allele"].to(device)
    for k in ("mutation_rate", "indel_count", "productive"):
        targets[k] = b[k].to(device)
    return batch_in, targets


def train_step(model, batch_in, targets, cfg, logvars, opt):
    from ..models.losses import hierarchical_loss
    model.train()
    out = model(batch_in)
    total, parts = hierarchical_loss(out, targets, cfg, logvars)
    opt.zero_grad()
    total.backward()
    opt.step()
    model.clamp_params()                       # analysis-head kernel constraints
    for uw in logvars.values():
        uw.apply_constraints()                 # Kendall log-var clamp
    return float(total.detach()), {k: float(v.detach()) for k, v in parts.items()}


def _stream_records(dataconfig, params, seed):
    p = dict(params)
    p.setdefault("invert_d_prob", 0.0)
    exp = build_experiment(dataconfig, p, allow_curatable=True)
    yield from exp.stream_records(n=None, seed=seed)


def train(model, reference_set, dataconfig, cfg, logvars, *, steps=2000, batch_size=32,
          lr=3e-4, progress=0.5, seed=0, device="cpu", log_every=50):
    from ..gym import Curriculum
    opt = torch.optim.AdamW(list(model.parameters()) + list(logvars.parameters()), lr=lr)
    stream = _stream_records(dataconfig, dict(Curriculum().params(progress)), seed)
    model.to(device)
    for step in range(1, steps + 1):
        records = list(itertools.islice(stream, batch_size))
        if len(records) < batch_size:
            break
        batch_in, targets = build_batch(records, reference_set, cfg, device)
        total, parts = train_step(model, batch_in, targets, cfg, logvars, opt)
        if step % log_every == 0:
            print(f"[{step}/{steps}] loss {total:.3f} "
                  + " ".join(f"{k}={v:.2f}" for k, v in parts.items()), flush=True)
    return model
