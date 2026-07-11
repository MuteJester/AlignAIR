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

from ..core.config import AlignAIRConfig
from .gym import build_experiment, build_targets, gym_collate
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
    # per-position edit-state labels (forward frame; the model canonicalizes internally) padded to L
    st = b["state_labels"]
    st = F.pad(st, (0, L - st.shape[1]), value=-100) if st.shape[1] < L else st[:, :L]
    targets["state_labels"] = st.to(device)
    return batch_in, targets


def train_step(model, batch_in, targets, cfg, logvars, opt):
    from ..core.losses import hierarchical_loss
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


def save_checkpoint(path, cfg, model, logvars, step, opt=None, *, dataconfigs=None, train_args=None):
    """Write a self-contained .alignair model file (weights + config + logvars + optimizer + RNG
    states + embedded dataconfig + training summary) — everything needed to CONTINUE training."""
    import random

    import numpy as np

    from .. import model_file as mf
    rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    ta = dict(train_args or {})
    training = {"steps": step, "batch_size": ta.get("batch_size", 0), "lr": ta.get("lr"),
                "progresses": ta.get("progresses"), "heavy_shm": ta.get("heavy_shm"),
                "short_boost": ta.get("short_boost"), "seed": ta.get("seed"), "train_args": ta}
    mf.save_model(path, model, dataconfigs=dataconfigs or [], training=training,
                  logvars=logvars, optimizer=opt, rng=rng)


def _ext(path: str) -> str:
    for e in (".alignair", ".pt"):
        if path.endswith(e):
            return e
    return ".alignair"


def _stem(base_path: str) -> str:
    e = _ext(base_path)
    return base_path[:-len(e)] if base_path.endswith(e) else base_path


def _step_of(path: str) -> int:
    m = re.search(r"\.step(\d+)\.(?:alignair|pt)$", path)
    return int(m.group(1)) if m else -1


def save_rotating(base_path, cfg, model, logvars, step, opt, keep=3, *, dataconfigs=None, train_args=None):
    """Write a full resumable .alignair into a rotating cycle of `keep` step-named files (so the
    3 latest states are always retained), and refresh `base_path` as a copy of the newest (for
    convenient --resume / benchmarking). A killed save can't corrupt older slots."""
    stem, ext = _stem(base_path), _ext(base_path)
    path = f"{stem}.step{step}{ext}"
    save_checkpoint(path, cfg, model, logvars, step, opt, dataconfigs=dataconfigs, train_args=train_args)
    shutil.copyfile(path, base_path)                      # base == newest
    for old in sorted(glob.glob(f"{stem}.step*{ext}"), key=_step_of)[:-keep]:
        if old != path:
            try:
                os.remove(old)
            except OSError:
                pass
    return path


def latest_checkpoint(base_path):
    """Highest-step rotating checkpoint (falls back to base_path), or None if nothing exists."""
    stem = _stem(base_path)
    files = glob.glob(f"{stem}.step*.alignair") + glob.glob(f"{stem}.step*.pt")
    if files:
        return max(files, key=_step_of)
    return base_path if os.path.exists(base_path) else None


def _stream_records(dataconfig, params, seed):
    p = dict(params)
    p.setdefault("invert_d_prob", 0.0)
    exp = build_experiment(dataconfig, p, allow_curatable=True)
    yield from exp.stream_records(n=None, seed=seed)


def _amplicon_specs(progresses, heavy_shm, short_boost=1, mutation_cap=None):
    """Training data mix as two orthogonal axes: corruption background (curriculum ``progress``) x
    amplicon mode (GenAIRR ``end_loss`` profile). All short-read shaping is native GenAIRR end-loss —
    no post-hoc cropping — so every V/D/J coordinate is engine-correct.

    - ``progresses`` streams: full-length reads at increasing corruption (rehearsal + noise).
    - amplicon streams: short reads by *one-sided* end-loss (never trimmed to empty) — V/FR-anchored
      keeps the 5' (trims 3'), J-anchored keeps the 3'/J (trims 5'); plus a bounded both-ends fragment.
    - heavy-SHM stream: full-length high-mutation (the heavy-SHM-V corner).

    ``short_boost`` repeats the amplicon block N times to up-weight short reads (as independent
    seeded streams) — e.g. a fine-tune that concentrates on short/cropped reads while retaining the
    full-length rehearsal streams. ``short_boost=1`` is the default balanced mix.
    """
    from .gym import Curriculum
    cur = Curriculum()
    specs = [dict(cur.params(pr)) for pr in progresses]        # full-length, corruption ramp
    bg = dict(cur.params(0.6))                                 # moderate corruption background

    def amp(e5, e3):
        s = dict(bg); s["end_loss_5"] = e5; s["end_loss_3"] = e3
        return s

    amplicons = [
        amp((0, 15), (150, 350)),                             # V / FR-anchored amplicon (keep 5')
        amp((150, 350), (0, 15)),                             # J-anchored amplicon (keep 3'/J)
        amp((70, 175), (70, 175)),                            # short fragment (both ends, bounded)
    ]
    specs += amplicons * max(1, int(short_boost))
    if heavy_shm > 0:
        hs = dict(cur.params(0.3)); hs["mutation_rate"] = heavy_shm   # heavy-SHM, full length
        specs.append(hs)
    if mutation_cap is not None:                    # loci without SHM (e.g. TCR): pin S5F mutation low
        for s in specs:
            s["mutation_rate"] = min(s["mutation_rate"], mutation_cap)
    return specs


def _mixed_stream(dataconfig, progresses, heavy_shm, seed, short_boost=1, mutation_cap=None):
    """Round-robin the amplicon-mode mix (:func:`_amplicon_specs`) so every batch spans full-length
    clean/hard + heavy-SHM + V-anchored/J-anchored/fragment short reads, all GenAIRR-native."""
    specs = _amplicon_specs(progresses, heavy_shm, short_boost, mutation_cap)
    streams = [_stream_records(dataconfig, s, seed + i) for i, s in enumerate(specs)]
    yield from itertools.chain.from_iterable(zip(*streams))


def eval_metrics(out: dict, targets: dict, cfg) -> dict:
    """Per-task quality on a batch (AlignAIR heads) — the ModelXRay ``task_eval`` for this model
    family: allele top-1-in-set, segmentation MAE, orientation accuracy, mutation/indel MAE,
    productivity accuracy."""
    m = {}
    for g in ["v", "j"] + (["d"] if cfg.has_d else []):
        top1 = out[f"{g}_allele"].argmax(-1)
        in_set = targets[f"{g}_allele"].gather(1, top1.unsqueeze(1)).squeeze(1)
        m[f"{g}_allele_top1"] = float((in_set > 0.5).float().mean())
        m[f"{g}_seg_mae"] = sum(
            float((out[f"{g}_{b}"].squeeze(-1) - targets[f"{g}_{b}"].squeeze(-1)).abs().mean())
            for b in ("start", "end")) / 2
    if "orientation_logits" in out and "orientation" in targets:
        m["orientation_acc"] = float((out["orientation_logits"].argmax(-1)
                                      == targets["orientation"]).float().mean())
    m["mutation_mae"] = float((out["mutation_rate"] - targets["mutation_rate"]).abs().mean())
    m["indel_mae"] = float((out["indel_count"] - targets["indel_count"]).abs().mean())
    m["productive_acc"] = float(((out["productive"] > 0.5) == (targets["productive"] > 0.5)).float().mean())
    if "state_logits" in out and "state_labels" in targets:
        pred, lab = out["state_logits"].argmax(-1), targets["state_labels"]
        valid = lab != -100
        m["state_acc"] = float((pred[valid] == lab[valid]).float().mean()) if valid.any() else float("nan")
        for name, idx in (("sub", 1), ("ins", 2), ("del", 3)):    # recall on the rare edit classes
            gt = valid & (lab == idx)
            m[f"state_{name}_recall"] = float((pred[gt] == idx).float().mean()) if gt.any() else float("nan")
    return m


def train(model, reference_set, dataconfig, cfg, logvars, *, steps=2000, batch_size=32,
          lr=3e-4, progresses=(0.3, 0.6, 0.9), heavy_shm=0.25, short_boost=1, seed=0, device="cpu",
          log_every=50, save_path=None, save_every=2000, resume_path=None,
          monitor_log=None, deep_every=500, mutation_cap=None):
    from .gym import Curriculum

    def _capped(pr):                               # curriculum params with SHM pinned low (TCR loci)
        s = dict(Curriculum().params(pr))
        if mutation_cap is not None:
            s["mutation_rate"] = min(s["mutation_rate"], mutation_cap)
        return s
    model.to(device)
    logvars.to(device)                         # Kendall log-vars must share the model's device
    opt = torch.optim.AdamW(list(model.parameters()) + list(logvars.parameters()), lr=lr)

    start = 0
    resume_from = latest_checkpoint(resume_path) if resume_path else None
    if resume_from:
        from .. import model_file as mf
        if mf.container.is_alignair_file(resume_from):
            ts = mf.load_training_state(resume_from, device=device)
            model.load_state_dict(ts.model.state_dict())
            if ts.logvars_state:
                logvars.load_state_dict(ts.logvars_state)
            if ts.optimizer_state:
                opt.load_state_dict(ts.optimizer_state)
                for grp in opt.param_groups:    # honor the CLI lr on resume (e.g. lower lr for a fine-tune)
                    grp["lr"] = lr
            start = ts.step
        else:
            ck = torch.load(resume_from, map_location=device)       # legacy .pt
            model.load_state_dict(ck["model"]); logvars.load_state_dict(ck["logvars"])
            if "optimizer" in ck:
                opt.load_state_dict(ck["optimizer"])
                for grp in opt.param_groups:
                    grp["lr"] = lr
            start = int(ck.get("step", 0))
        print(f"RESUMED from {resume_from} at step {start} (lr={lr}, short_boost={short_boost})", flush=True)

    _train_args = {"lr": lr, "batch_size": batch_size, "progresses": list(progresses),
                   "heavy_shm": heavy_shm, "short_boost": short_boost, "seed": seed, "steps": steps,
                   "mutation_cap": mutation_cap}
    _dcs = [dataconfig]

    # ModelXRay: a read-only training X-ray over a fixed held-out probe batch
    xray = probe_input = None
    if monitor_log:
        from ..xray import ModelXRay
        ev = list(itertools.islice(
            _stream_records(dataconfig, _capped(max(progresses)), seed + 99991),
            batch_size))
        probe_input, probe_targets = build_batch(ev, reference_set, cfg, device)
        from ..core.losses import hierarchical_loss

        def task_eval(m, inp, _tg=probe_targets, _cfg=cfg):
            return eval_metrics(m(inp), _tg, _cfg)

        def task_losses(m, inp, _tg=probe_targets, _cfg=cfg, _lv=logvars):
            return hierarchical_loss(m(inp), _tg, _cfg, _lv)[1]      # per-task loss tensors

        xray = ModelXRay(model, lr=lr, log_path=monitor_log, deep_every=deep_every,
                         uncertainty=logvars, task_eval=task_eval, task_losses=task_losses,
                         shared_module="embedding")

    stream = (_mixed_stream(dataconfig, progresses, heavy_shm, seed + start, short_boost, mutation_cap)
              if len(progresses) > 1 or heavy_shm > 0 or short_boost > 1
              else _stream_records(dataconfig, _capped(progresses[0]), seed + start))
    for step in range(start + 1, steps + 1):
        records = list(itertools.islice(stream, batch_size))
        if len(records) < batch_size:
            break
        batch_in, targets = build_batch(records, reference_set, cfg, device)
        total, parts = train_step(model, batch_in, targets, cfg, logvars, opt)
        if step % log_every == 0:
            line = (f"[{step}/{steps}] loss {total:.3f} "
                    + " ".join(f"{k}={v:.2f}" for k, v in parts.items()))
            if xray is not None:                   # reads this step's grads (pre next zero_grad)
                rec = xray.observe(step, total, parts, probe_input=probe_input)
                line += "  " + xray.summary_line(rec)
            print(line, flush=True)
        if save_path and step % save_every == 0:
            save_rotating(save_path, cfg, model, logvars, step, opt, dataconfigs=_dcs, train_args=_train_args)
    if save_path:
        save_rotating(save_path, cfg, model, logvars, steps, opt, dataconfigs=_dcs, train_args=_train_args)
    if xray is not None:
        xray.close()
    return model
