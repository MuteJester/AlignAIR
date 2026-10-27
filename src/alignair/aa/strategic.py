"""Categorical protein training with explicit curriculum and acceptance policies.

Shared sample preparation maps V/J protein classes and optional D alleles. This runner retains
nonproductive/stop-containing reads and applies length-preserving substitutions before batching.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.augmentation.protein import aa_substitute
from ..core import prepare_model
from ..reference import build_protein_reference
from ..data.adapters import CategoricalBatchAdapter
from ..data.sampling import ProteinSampleBuilder, ProteinSamplePolicy
from ..tasks import build_tasks
from ..train.engine import TrainingEngine
from ..train.loop import training_iterations
from ..train.checkpoints import save_snapshot as _save
from ..data.tokenizer import AAProteinTokenizer
from .train_gym import _usable_stream


def strategic_setup(dc_name: str = "HUMAN_IGH_OGRDB", max_seq_length: int = 192, has_d: bool = True):
    reference = build_protein_reference(dc_name)
    setup = prepare_model(reference, input_space="protein", label_space="protein_classes",
                          max_seq_length=max_seq_length, has_d=has_d)
    d_index = dict(setup.reference.gene("D").index) if setup.config.has_d else {}
    meta = {**reference.metadata(), "n_d_alleles": len(d_index), "d_index": d_index}
    return setup.build_model(), setup.config, reference.class_maps, meta


def _sample_stream(dc_name, class_maps, d_index, tok, max_len, sub_prob, seed, rng):
    """Real gym records -> class-target samples, with length-preserving substitution augmentation."""
    builder = ProteinSampleBuilder(class_maps, tok, max_len, d_index,
                                   policy=ProteinSamplePolicy(require_productive=False, allow_stops=True))
    for rec in _usable_stream(dc_name, (0.3, 0.6, 0.9), 0.25, seed, 1, None):
        s = builder(rec).sample
        if s is None:
            continue
        if sub_prob and rng.random() < sub_prob:
            enc = tok.encode(aa_substitute(rec["sequence_aa"], float(rng.uniform(0.02, 0.10)), rng))
            if len(enc) == len(s["tokens"]):               # substitution preserves length -> boundaries valid
                s = {**s, "tokens": enc}
        yield s


def strategic_train(model, cfg, class_maps, dc_name="HUMAN_IGH_OGRDB", *, d_index=None, steps=50000,
                    batch_size=64, lr=3e-4, sub_prob=0.5, seed=0, device="cpu", log_every=250,
                    grad_clip=1.0, save_path=None, save_every=5000):
    tok = AAProteinTokenizer()
    rng = np.random.default_rng(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    d_index = d_index or {}
    genes = ("v", "d", "j") if cfg.has_d else ("v", "j")
    tasks = build_tasks(model.cfg, genes=genes)
    engine = TrainingEngine(model, opt, tasks=tasks, grad_clip=grad_clip)
    stream = _sample_stream(dc_name, class_maps, d_index, tok, cfg.max_seq_length, sub_prob, seed, rng)

    batch_adapter = CategoricalBatchAdapter.from_config(cfg, device=device)
    history = []
    for iteration in training_iterations(engine, stream, batch_adapter, stop_step=steps, batch_size=batch_size):
        step, result = iteration.step, iteration.result
        history.append({"step": step, "total": result.total, **result.parts})
        if log_every and step % log_every == 0:
            h = history[-1]
            print(f"[{step}/{steps}] loss {h['total']:.3f} seg {h['segmentation']:.3f} "
                  f"cls {h['classification']:.3f}", flush=True)
        if save_path and save_every and step % save_every == 0:
            _save(save_path, model, cfg, step)
    if save_path:
        _save(save_path, model, cfg, history[-1]["step"] if history else 0)
    return history
