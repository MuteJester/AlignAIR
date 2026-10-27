"""Stage-1 coordinate-aware truncation training.

Augmentation mix per read: full-length rehearsal, N/C/internal crops (coordinate-aware — boundaries
shifted+clipped into crop coords, absent genes masked, partially-observed V/J supervised against the
compatible protein-class SET), and length-preserving substitutions / I/L ambiguity. D stays an
allele posterior (no protein-equivalence merging — its observability is frame/trim/junction-dependent):
fully observed -> singleton true allele; partial/absent -> masked.

Resumability: shared checkpoint I/O retains optimizer, scheduler, process RNG, and provenance.
The runner deterministically re-seeds the source stream and local augmentation generator at resume;
this is statistical continuation, not a replay of the exact next uninterrupted batch."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import dataclasses
import hashlib
import subprocess

import numpy as np
import torch

from ..errors import ConfigurationError
from ..data.augmentation import ProteinCropAugmenter, ProteinCropPolicy, SAMPLE_POLICY_ID
from ..data.adapters import CategoricalBatchAdapter
from ..tasks import build_tasks
from ..reference.protein import build_protein_reference, prototypes_by_class
from ..train.checkpoints import (load_research_checkpoint as load_checkpoint,
                                  save_research_checkpoint as save_checkpoint,
                                  research_checkpoint_rng, save_snapshot as _save)
from ..train.random_state import restore_rng_state
from ..core.runtime import evaluation_mode
from ..train.setup import build_training_engine
from ..train.loop import training_iterations
from ..data.sampling import ProteinSampleBuilder, ProteinSamplePolicy
from ..data.tokenizer import AAProteinTokenizer
from ..train.gym.streams import CurriculumStreamPolicy, training_records


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:                                        # noqa: BLE001 - provenance must not break a run
        return "unknown"


def _sha256_file(path) -> str | None:
    if not path:
        return None
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def validate_stage1_resume(checkpoint):
    """Reject optimizer continuation when the checkpoint's target policy is incompatible."""
    if checkpoint is not None and checkpoint["optimizer"] is not None:
        provenance = checkpoint.get("provenance")
        if not isinstance(provenance, Mapping) or provenance.get("sample_policy") != SAMPLE_POLICY_ID:
            raise ConfigurationError("checkpoint crop-label policy differs from this trainer",
                                     hint="Use init_path for a weights-only warm start under the new label policy.")


def stage1_train(model, cfg, class_maps, meta, dc_name="HUMAN_IGH_OGRDB", *, steps=40000, batch_size=64,
                 lr=3e-4, p_crop=0.4, p_sub=0.4, p_il=0.15, seed=0, device="cpu", log_every=250,
                 grad_clip=1.0, save_path=None, save_every=5000, init_path=None,
                 resume_path=None, base_lr=1e-4, lr_min=1e-5, total_budget=None,
                 val_every=0, val_fn=None, ckpt_dir=None, ckpt_steps=(), plateau_k=0,
                 protein_reference=None):
    if init_path and resume_path:
        raise ConfigurationError("choose init_path for warm starts or resume_path for continuation")
    ck = load_checkpoint(resume_path) if resume_path else None
    validate_stage1_resume(ck)
    reference = protein_reference if protein_reference is not None else build_protein_reference(dc_name)
    if init_path:                                            # warm-start (fine-tune a passing baseline)
        model.load_state_dict(load_checkpoint(init_path)["model"])
    tok = AAProteinTokenizer()
    d_index = meta.get("d_index") if cfg.has_d else None
    if cfg.has_d and not d_index:
        raise ConfigurationError("D-enabled crop training requires an explicit D allele vocabulary")
    sample_builder = ProteinSampleBuilder(class_maps, tok, cfg.max_seq_length, d_index,
                                         policy=ProteinSamplePolicy(require_productive=False, allow_stops=True))
    model.train()
    genes = tuple(spec.name for spec in cfg.gene_specs)
    tasks = build_tasks(model.cfg, genes=genes)
    engine = build_training_engine(
        model, tasks=tasks, device=device, lr=base_lr, grad_clip=grad_clip,
        scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(total_budget or steps), eta_min=lr_min),
    )
    opt, sch = engine.optimizer, engine.scheduler

    start_step = 0
    if resume_path:
        model.load_state_dict(ck["model"])
        if ck["optimizer"] is not None:                       # full-state checkpoint -> faithful continuation
            opt.load_state_dict(ck["optimizer"])
            sch.load_state_dict(ck["scheduler"])
            restore_rng_state(research_checkpoint_rng(ck))
            start_step = ck["step"]
        else:                                                 # old {model,cfg,step} incumbent -> warm-start only
            start_step = ck.get("step") or 0
            print(f"[resume] '{resume_path}' has no optimizer state (old-format checkpoint); "
                  f"warm-starting weights only from step {start_step} with a FRESH optimizer/scheduler "
                  f"— this is the one unavoidable non-faithful hop.", flush=True)

    rng = np.random.default_rng([seed, start_step])           # deterministic stream re-seed at the resume point
    stream_seed = seed + start_step                            # gym/corpus stream must also re-seed at the
                                                                # resume point (mirrors trainer.py's seed+start),
                                                                # else every resume replays the corpus from record 0

    augmenter = ProteinCropAugmenter(reference, sample_builder, rng, ProteinCropPolicy(p_crop, p_sub, p_il))

    def gen():
        records = training_records((dc_name,), policy=CurriculumStreamPolicy(mode="amplicon"),
                                   seed=stream_seed, translated_only=True)
        for rec in records:
            s = augmenter(rec).sample
            if s is not None:
                yield s
    stream = gen()

    batch_adapter = CategoricalBatchAdapter.from_config(cfg, device=device)
    history = []
    val_history = []
    for iteration in training_iterations(engine, stream, batch_adapter, start_step=start_step,
                                         stop_step=start_step + steps, batch_size=batch_size):
        step, result, batch = iteration.step, iteration.result, iteration.records
        class_support = {gene: sum(bool(row[f"{gene}_class_supervise"]) for row in batch) for gene in genes}
        class_reasons = {gene: dict(Counter(row[f"{gene}_class_reason"] for row in batch))
                         for gene in genes}
        history.append({"step": step, "total": result.total, "lr": sch.get_last_lr()[0],
                        "class_support": class_support, "class_reasons": class_reasons, **result.parts})
        if log_every and step % log_every == 0:
            h = history[-1]
            print(f"[{step}/{start_step + steps}] loss {h['total']:.3f} seg {h['segmentation']:.3f} "
                  f"cls {h['classification']:.3f} lr {h['lr']:.2e}", flush=True)
        if ckpt_dir and step in ckpt_steps:
            provenance = {
                "git_sha": _git_sha(), "base_model_sha256": _sha256_file(resume_path),
                "seed": seed, "dc_name": dc_name, "created_step": step, "sample_policy": SAMPLE_POLICY_ID,
                "hyperparams": {"base_lr": base_lr, "lr_min": lr_min, "total_budget": total_budget,
                                "batch_size": batch_size, "p_crop": p_crop, "p_sub": p_sub, "p_il": p_il},
            }
            save_checkpoint(f"{ckpt_dir}/model_cont_step{step}.pt", model=model,
                            cfg=dataclasses.asdict(cfg), step=step, optimizer=opt, scheduler=sch,
                            lr=sch.get_last_lr()[0], history=history, provenance=provenance)
        if val_every and val_fn and step % val_every == 0:
            with evaluation_mode(model):
                v = val_fn(model, step)
            val_history.append(v)
            objectives = [r.get("objective") for r in val_history[-plateau_k:]] if plateau_k else []
            if plateau_k and len(objectives) >= plateau_k and all(o is not None for o in objectives) \
                    and all(objectives[i] <= objectives[i - 1] for i in range(1, len(objectives))):
                print(f"[plateau] validation objective has not improved over the last {plateau_k} "
                      f"evaluations (step {step}).", flush=True)
        if save_path and save_every and step % save_every == 0:
            _save(save_path, model, cfg, step)
    if save_path:
        _save(save_path, model, cfg, history[-1]["step"] if history else start_step)
    return history
