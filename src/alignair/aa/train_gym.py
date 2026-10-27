"""Train the amino-acid AlignAIR with the SAME scheme as the nucleotide model.

The model (real ``AlignAIR`` with ``use_orientation=False``), loss (``hierarchical_loss``), optimizer
step (``train_step``), and data stream (the gym's ``_mixed_stream`` curriculum) are all reused. The
only differences are the AA input tokens and AA-space coordinates (see ``data.adapters``).
Records without a translation/frame or with inconsistent translation length are skipped."""
from __future__ import annotations

import torch

from ..core.config import AlignAIRConfig
from ..tasks import build_tasks
from ..train.engine import TrainingEngine
from ..train.checkpoints import save_snapshot as _save
from ..core.model import AlignAIR
from ..reference.reference_set import ReferenceSet
from ..data.adapters import build_protein_allele_batch as build_batch_aa
from ..data.projection import translated_record_usable as aa_usable
from ..data.tokenizer import AAProteinTokenizer


def setup_aa(dc_name: str = "HUMAN_IGH_OGRDB", max_seq_length: int = 192):
    """Build (model, reference_set, cfg, logvars) for the AA model over a GenAIRR dataconfig.

    The config derives the FULL nucleotide allele counts from the reference (198/33/7 for IGH) — the
    allele heads predict nucleotide alleles, so synonymous ambiguity must emerge on its own."""
    import GenAIRR as ga
    dc = getattr(ga, dc_name)
    ref = ReferenceSet.from_dataconfigs(dc)
    cfg = AlignAIRConfig.from_dataconfigs(dc, max_seq_length=max_seq_length,
                                          vocab_size=23, use_orientation=False)
    model = AlignAIR(cfg)
    logvars = build_tasks(cfg).make_weights()
    return model, ref, cfg, logvars


def _usable_stream(dc_name, progresses, heavy_shm, seed, short_boost, mutation_cap):
    """The gym's mixed-amplicon curriculum stream, filtered by shared translation eligibility."""
    import GenAIRR as ga
    from ..train.trainer import _mixed_stream
    dc = getattr(ga, dc_name)
    for rec in _mixed_stream(dc, progresses, heavy_shm, seed, short_boost, mutation_cap):
        if aa_usable(rec):
            yield rec


def _next_batch(stream, n):
    recs = []
    for rec in stream:
        recs.append(rec)
        if len(recs) >= n:
            break
    return recs


def train_aa(model, reference_set, dc_name, cfg, logvars, *, steps=2000, batch_size=64, lr=3e-4,
             progresses=(0.3, 0.6, 0.9), heavy_shm=0.25, short_boost=1, mutation_cap=None,
             seed=0, device="cpu", log_every=50, grad_clip=1.0, save_path=None, save_every=2000):
    """Train with the gym curriculum; returns a per-step history of losses."""
    model.to(device)
    logvars.to(device)
    tasks = build_tasks(cfg)
    opt = torch.optim.AdamW(list(model.parameters()) + list(logvars.parameters()), lr=lr)
    engine = TrainingEngine(model, opt, tasks=tasks, weights=logvars, grad_clip=grad_clip)
    stream = _usable_stream(dc_name, progresses, heavy_shm, seed, short_boost, mutation_cap)
    tok = AAProteinTokenizer()

    history = []
    for step in range(1, steps + 1):
        recs = _next_batch(stream, batch_size)
        if len(recs) < batch_size:
            break
        bi, tg = build_batch_aa(recs, reference_set, cfg, device, tok)
        total, parts, gnorm = engine.step((bi, tg), step=step)
        history.append({"step": step, "total": total, "gnorm": gnorm, "parts": parts})
        if log_every and step % log_every == 0:
            line = (f"[{step}/{steps}] loss {total:.3f} gnorm {gnorm:.2f} "
                    + " ".join(f"{k}={v:.2f}" for k, v in parts.items()))
            print(line, flush=True)
        if save_path and save_every and step % save_every == 0:
            _save(save_path, model, cfg, step)
    if save_path:
        _save(save_path, model, cfg, history[-1]["step"] if history else 0)
    return history
