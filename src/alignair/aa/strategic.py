"""Strategic AA path (b): explicit protein-equivalence CLASS targets, on the REAL gym distribution,
with AA-native augmentation, trained longer. Reuses Phase-1's softmax class-head model (`AAAlignAIR`)
and loss (`aa_loss`), fed by the real curriculum gym (`train_gym._usable_stream`) via the class-target
adapter (`data.record_to_sample`). Substitution augmentation is length-preserving, so boundary targets
stay valid; truncation/indel/IL are evaluation strata (see `augment.py`)."""
from __future__ import annotations

import itertools

import numpy as np
import torch

from .augment import aa_substitute
from .config import aa_igh_config
from .data import build_class_maps, collate_aa, record_to_sample
from .loss import aa_loss
from .model import AAAlignAIR
from .tokenizer import AAProteinTokenizer
from .train_gym import _usable_stream


def strategic_setup(dc_name: str = "HUMAN_IGH_OGRDB", max_seq_length: int = 192):
    class_maps, meta = build_class_maps(dc_name)
    cfg = aa_igh_config(meta["n_v_classes"], meta["n_j_classes"], max_seq_length)
    return AAAlignAIR(cfg), cfg, class_maps, meta


def _sample_stream(dc_name, class_maps, tok, max_len, sub_prob, seed, rng):
    """Real gym records -> class-target samples, with length-preserving substitution augmentation."""
    for rec in _usable_stream(dc_name, (0.3, 0.6, 0.9), 0.25, seed, 1, None):
        s, why = record_to_sample(rec, class_maps, tok, max_len)
        if s is None:
            continue
        if sub_prob and rng.random() < sub_prob:
            aug = aa_substitute(rec["sequence_aa"], float(rng.uniform(0.02, 0.10)), rng)
            enc = tok.encode(aug)
            if len(enc) == len(s["tokens"]):               # substitution preserves length -> boundaries valid
                s = {**s, "tokens": enc}
        yield s


def strategic_train(model, cfg, class_maps, dc_name="HUMAN_IGH_OGRDB", *, steps=50000, batch_size=64,
                    lr=3e-4, sub_prob=0.5, seed=0, device="cpu", log_every=250, grad_clip=1.0,
                    save_path=None, save_every=5000):
    tok = AAProteinTokenizer()
    rng = np.random.default_rng(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    stream = _sample_stream(dc_name, class_maps, tok, cfg.max_seq_length, sub_prob, seed, rng)

    history = []
    for step in range(1, steps + 1):
        batch = list(itertools.islice(stream, batch_size))
        if len(batch) < batch_size:
            break
        bi, tg = collate_aa(batch, cfg.max_seq_length, device)
        out = model(bi)
        total, parts = aa_loss(out, tg)
        opt.zero_grad()
        total.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        history.append({"step": step, "total": float(total.detach()),
                        **{k: float(v.detach()) for k, v in parts.items()}})
        if log_every and step % log_every == 0:
            h = history[-1]
            print(f"[{step}/{steps}] loss {h['total']:.3f} seg {h['segmentation']:.3f} "
                  f"cls {h['classification']:.3f}", flush=True)
        if save_path and save_every and step % save_every == 0:
            _save(save_path, model, cfg, step)
    if save_path:
        _save(save_path, model, cfg, history[-1]["step"] if history else 0)
    return history


def _save(path, model, cfg, step):
    import dataclasses
    torch.save({"model": model.state_dict(), "cfg": dataclasses.asdict(cfg), "step": step}, path)
