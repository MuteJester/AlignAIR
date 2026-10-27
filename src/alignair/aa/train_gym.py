"""Train the amino-acid AlignAIR with the SAME scheme as the nucleotide model.

The model (real ``AlignAIR`` with ``use_orientation=False``), loss (``hierarchical_loss``), optimizer
step (``train_step``), and data stream (the shared gym curriculum) are all reused. The
only differences are the AA input tokens and AA-space coordinates (see ``data.adapters``).
Records without a translation/frame or with inconsistent translation length are skipped."""
from __future__ import annotations

import torch

from ..core import prepare_model
from ..tasks import build_tasks
from ..train.engine import TrainingEngine
from ..train.loop import training_iterations
from ..train.checkpoints import save_snapshot as _save
from ..reference.reference_set import ReferenceSet
from ..data.adapters import ProteinAlleleBatchAdapter
from ..train.gym.streams import CurriculumStreamPolicy, training_records
from ..data.tokenizer import AAProteinTokenizer


def setup_aa(dc_name: str = "HUMAN_IGH_OGRDB", max_seq_length: int = 192):
    """Build (model, reference_set, cfg, logvars) for the AA model over a GenAIRR dataconfig.

    The config derives the FULL nucleotide allele counts from the reference (198/33/7 for IGH) — the
    allele heads predict nucleotide alleles, so synonymous ambiguity must emerge on its own."""
    import GenAIRR as ga
    dc = getattr(ga, dc_name)
    ref = ReferenceSet.from_dataconfigs(dc)
    setup = prepare_model(ref, input_space="protein", max_seq_length=max_seq_length)
    return setup.build_model(), setup.reference, setup.config, build_tasks(setup.config).make_weights()


def train_aa(model, reference_set, dc_name, cfg, logvars, *, steps=2000, batch_size=64, lr=3e-4,
             progresses=(0.3, 0.6, 0.9), heavy_shm=0.25, short_boost=1, mutation_cap=None,
             seed=0, device="cpu", log_every=50, grad_clip=1.0, save_path=None, save_every=2000):
    """Train with the gym curriculum; returns a per-step history of losses."""
    model.to(device)
    logvars.to(device)
    tasks = build_tasks(cfg)
    opt = torch.optim.AdamW(list(model.parameters()) + list(logvars.parameters()), lr=lr)
    engine = TrainingEngine(model, opt, tasks=tasks, weights=logvars, grad_clip=grad_clip)
    policy = CurriculumStreamPolicy(progresses, heavy_shm, short_boost, mutation_cap, mode="amplicon")
    stream = training_records((dc_name,), policy=policy, seed=seed, translated_only=True)
    tok = AAProteinTokenizer()

    batch_adapter = ProteinAlleleBatchAdapter(reference_set, cfg, device, tok)
    history = []
    for iteration in training_iterations(engine, stream, batch_adapter, stop_step=steps, batch_size=batch_size):
        step = iteration.step
        total, parts, gnorm = iteration.result
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
