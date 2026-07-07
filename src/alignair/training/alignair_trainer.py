"""Training loop for the faithful AlignAIR model on the GenAIRR gym stream.

``build_batch`` adapts the reusable ``gym_collate`` output (tokens, query coords, allele multi-hot,
mutation/indel/productive) into the model's input + target dicts. ``train_step`` runs one
optimizer step and applies the TF weight *constraints* (analysis-head kernel clamps + Kendall
log-var clamps) after ``optimizer.step()``.
"""
from __future__ import annotations

import glob
import itertools
import os
import re
import shutil

import torch
import torch.nn.functional as F

from ..config.alignair_config import AlignAIRConfig
from ..gym import build_experiment, build_targets, gym_collate
from ..nn.heads.orientation import NUM_ORIENTATIONS, apply_orientation


def build_batch(records, reference_set, cfg: AlignAIRConfig, device: str = "cpu",
                augment_orientation: bool = True):
    """(records, reference) -> (model_input, targets), tokens padded/truncated to max_seq_length.

    The gym stream is forward-only, so ``augment_orientation`` applies a random orientation (one of
    4) to each read's tokens and supplies the label — the model learns to detect and self-correct it,
    while the segmentation/allele targets stay in the forward frame (correction re-canonicalizes).
    """
    bundles = [build_targets(r, reference_set, cfg.has_d) for r in records]
    b = gym_collate(bundles, reference_set, cfg.has_d)
    L = cfg.max_seq_length
    tok = b["tokens"]
    tok = F.pad(tok, (0, L - tok.shape[1])) if tok.shape[1] < L else tok[:, :L]

    if augment_orientation:
        orient = torch.randint(0, NUM_ORIENTATIONS, (tok.shape[0],))
        tok = apply_orientation(tok, tok != 0, orient)
    else:
        orient = torch.zeros(tok.shape[0], dtype=torch.long)
    batch_in = {"tokenized_sequence": tok.to(device), "orientation": orient.to(device)}

    genes = ["v", "j"] + (["d"] if cfg.has_d else [])
    targets: dict = {}
    for g in genes:
        targets[f"{g}_start"] = b[f"{g}_start"].float().unsqueeze(-1).to(device)
        targets[f"{g}_end"] = b[f"{g}_end"].float().unsqueeze(-1).to(device)
        targets[f"{g}_allele"] = b[f"{g}_allele"].to(device)
    for k in ("mutation_rate", "indel_count", "productive"):
        targets[k] = b[k].to(device)
    targets["orientation"] = orient.to(device)
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


def save_checkpoint(path, cfg, model, logvars, step, opt=None):
    """Full, self-sufficient checkpoint: config + model + Kendall log-vars + optimizer + step —
    everything needed to CONTINUE training from this exact state."""
    ck = {"config": cfg.__dict__, "model": model.state_dict(),
          "logvars": logvars.state_dict(), "step": step}
    if opt is not None:
        ck["optimizer"] = opt.state_dict()
    torch.save(ck, path)


def _step_of(path: str) -> int:
    m = re.search(r"\.step(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def _stem(base_path: str) -> str:
    return base_path[:-3] if base_path.endswith(".pt") else base_path


def save_rotating(base_path, cfg, model, logvars, step, opt, keep=3):
    """Write a full resumable checkpoint into a rotating cycle of `keep` step-named files (so the
    3 latest states are always retained), and refresh `base_path` as a copy of the newest (for
    convenient --resume / benchmarking). A killed save can't corrupt older slots."""
    stem = _stem(base_path)
    path = f"{stem}.step{step}.pt"
    save_checkpoint(path, cfg, model, logvars, step, opt)
    shutil.copyfile(path, base_path)                      # base == newest
    for old in sorted(glob.glob(f"{stem}.step*.pt"), key=_step_of)[:-keep]:
        if old != path:
            try:
                os.remove(old)
            except OSError:
                pass
    return path


def latest_checkpoint(base_path):
    """Highest-step rotating checkpoint (falls back to base_path), or None if nothing exists."""
    files = glob.glob(f"{_stem(base_path)}.step*.pt")
    if files:
        return max(files, key=_step_of)
    return base_path if os.path.exists(base_path) else None


def _stream_records(dataconfig, params, seed):
    p = dict(params)
    p.setdefault("invert_d_prob", 0.0)
    exp = build_experiment(dataconfig, p, allow_curatable=True)
    yield from exp.stream_records(n=None, seed=seed)


def _mixed_stream(dataconfig, progresses, heavy_shm, seed):
    """Round-robin over difficulty levels so a long run trains on clean + hard + heavy-SHM +
    cropped/fragment reads (the strata the 20k run was weak on, incl. orientation-on-fragments)."""
    from ..gym import Curriculum
    specs = [dict(Curriculum().params(pr)) for pr in progresses]
    if heavy_shm > 0:
        hs = dict(Curriculum().params(1.0)); hs["mutation_rate"] = heavy_shm
        specs.append(hs)
    streams = [_stream_records(dataconfig, s, seed + i) for i, s in enumerate(specs)]
    yield from itertools.chain.from_iterable(zip(*streams))


def train(model, reference_set, dataconfig, cfg, logvars, *, steps=2000, batch_size=32,
          lr=3e-4, progresses=(0.3, 0.6, 0.9), heavy_shm=0.25, seed=0, device="cpu",
          log_every=50, save_path=None, save_every=2000, resume_path=None,
          monitor_log=None, deep_every=500):
    from ..gym import Curriculum
    model.to(device)
    logvars.to(device)                         # Kendall log-vars must share the model's device
    opt = torch.optim.AdamW(list(model.parameters()) + list(logvars.parameters()), lr=lr)

    start = 0
    resume_from = latest_checkpoint(resume_path) if resume_path else None
    if resume_from:
        ck = torch.load(resume_from, map_location=device)
        model.load_state_dict(ck["model"]); logvars.load_state_dict(ck["logvars"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start = int(ck.get("step", 0))
        print(f"RESUMED from {resume_from} at step {start}", flush=True)

    # fixed held-out batch for deep diagnostics (per-task eval + activation health)
    monitor = eval_batch = None
    if monitor_log:
        from .diagnostics import TrainingMonitor
        monitor = TrainingMonitor(lr, monitor_log, deep_every)
        ev = list(itertools.islice(
            _stream_records(dataconfig, dict(Curriculum().params(max(progresses))), seed + 99991),
            batch_size))
        bi, tg = build_batch(ev, reference_set, cfg, device)
        eval_batch = {"input": bi, "targets": tg}

    stream = (_mixed_stream(dataconfig, progresses, heavy_shm, seed + start)
              if len(progresses) > 1 or heavy_shm > 0
              else _stream_records(dataconfig, dict(Curriculum().params(progresses[0])), seed + start))
    for step in range(start + 1, steps + 1):
        records = list(itertools.islice(stream, batch_size))
        if len(records) < batch_size:
            break
        batch_in, targets = build_batch(records, reference_set, cfg, device)
        total, parts = train_step(model, batch_in, targets, cfg, logvars, opt)
        if step % log_every == 0:
            line = (f"[{step}/{steps}] loss {total:.3f} "
                    + " ".join(f"{k}={v:.2f}" for k, v in parts.items()))
            if monitor is not None:                # reads this step's grads (pre next zero_grad)
                rec = monitor.observe(model, logvars, total, parts, step, eval_batch, cfg)
                line += "  " + monitor.summary_line(rec)
            print(line, flush=True)
        if save_path and step % save_every == 0:
            save_rotating(save_path, cfg, model, logvars, step, opt)
    if save_path:
        save_rotating(save_path, cfg, model, logvars, steps, opt)
    if monitor is not None:
        monitor.close()
    return model
