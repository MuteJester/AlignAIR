"""Training loop for the faithful AlignAIR model on the GenAIRR gym stream.

``build_batch`` is a compatibility import of the shared data adapter. ``train_step`` runs one
optimizer step and applies the TF weight *constraints* (analysis-head kernel clamps + Kendall
log-var clamps) after ``optimizer.step()``.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from ..errors import ConfigurationError, error_context
from ..data.adapters import build_nucleotide_batch as build_batch
from ..data.training import prepare_training_data
from ..core.setup import ModelSetup
from ..reference.protein import build_protein_reference
from ..tasks import TrainingTasks, build_tasks
from .gym.streams import CurriculumStreamPolicy, training_records, effective_mutation_caps
from .engine import train_step, validate
from .setup import build_training_engine
from .loop import training_iterations
from .checkpoints import (save_checkpoint, save_rotating, latest_checkpoint, load_training_checkpoint,
                          validate_continuation, _stem, _ext)
from .random_state import restore_rng_state as _restore_rng
from .schedules import CosineSchedule
from .study import TrainingStudy, StudySession


def eval_metrics(out: dict, targets: dict, cfg) -> dict:
    """Compatibility entry for the shared alignment task metrics."""
    return build_tasks(cfg).metrics(out, targets)


@dataclass(frozen=True)
class TrainingOutcome:
    """This invocation's completed updates and best checkpoint, never inferred from old files."""
    model: object
    steps: int
    best_model_path: str | None
    checkpoints: dict = field(default_factory=dict)
    validations: tuple = ()


def train(model, reference_set, dataconfigs, cfg, logvars=None, *, steps=2000, batch_size=32,
          lr=3e-4, progresses=(0.3, 0.6, 0.9), heavy_shm=0.25, short_boost=1, seed=0, device="cpu",
          log_every=50, save_path=None, save_every=2000, resume_path=None,
          monitor_log=None, deep_every=500, mutation_cap=None, init_from=None, init_trust_pickle=False,
          grad_clip=None, val_every=0, val_batches=4, resume_trust_pickle=False,
          tasks: TrainingTasks | None = None, model_setup: ModelSetup | None = None, on_iteration=None,
          augmentation=None, schedule=None, study=None, start_step=0):
    from .guards import validate_training_request
    tasks = build_tasks(cfg) if tasks is None else tasks
    dataconfigs = list(dataconfigs) if isinstance(dataconfigs, (list, tuple)) else [dataconfigs]
    # preflight: fail fast on a bad request rather than after a long run
    validate_training_request(steps=steps, batch_size=batch_size, lr=lr,
                              max_seq_length=cfg.max_seq_length, reference=reference_set,
                              progresses=progresses, heavy_shm=heavy_shm, short_boost=short_boost,
                              grad_clip=grad_clip)
    if on_iteration is not None and not callable(on_iteration):
        raise ConfigurationError('on_iteration must be callable or None')
    if (init_from and resume_path or isinstance(start_step, bool) or not isinstance(start_step, int)
            or start_step < 0 or start_step and not init_from):
        raise ConfigurationError('choose resume_path or init_from; a warm-start offset requires init_from')
    if schedule is not None:
        if not isinstance(schedule, CosineSchedule):
            raise ConfigurationError('schedule must be a CosineSchedule or None')
        schedule.validate(lr)
    if study is not None and not isinstance(study, TrainingStudy):
        raise ConfigurationError('study must be a TrainingStudy or None')

    setup = model_setup or ModelSetup(reference_set, cfg, cfg.input_space, cfg.label_space,
        build_protein_reference(reference_set) if cfg.label_space == 'protein_classes' else None)
    if setup.reference is not reference_set or setup.config != cfg:
        raise ConfigurationError('training setup must bind this model configuration and reference')
    data = prepare_training_data(setup, device=device, augmentation=augmentation)
    augmentation_metadata = augmentation.to_dict() if augmentation is not None else None
    schedule_metadata = schedule.to_dict() if schedule is not None else None
    schema = setup.label_schema()
    source_policy = CurriculumStreamPolicy(progresses, heavy_shm, short_boost, mutation_cap,
                                           mode='amplicon' if cfg.input_space == 'protein' else 'auto')

    def _build_stream(seed_value, *, augment=False, start_step=0):
        records = training_records(dataconfigs, policy=source_policy, seed=seed_value,
                                   translated_only=data.translated_only)
        return data.prepare_records(records, augment=augment, seed=seed if augment else seed_value,
                                    start_step=start_step)

    engine = build_training_engine(model, tasks=tasks, weights=logvars, device=device,
                                   lr=lr, grad_clip=grad_clip,
                                   scheduler_factory=schedule.build if schedule is not None else None)
    opt, logvars = engine.optimizer, engine.weights

    start = start_step
    resume_from = latest_checkpoint(resume_path) if resume_path else None
    if resume_path and resume_from is None:
        raise ConfigurationError('resume checkpoint does not exist', context={'path': str(resume_path)})
    if resume_from:
        state = load_training_checkpoint(resume_from, device=device, trust_pickle=resume_trust_pickle)
        validate_continuation(state, cfg, schema, augmentation=augmentation_metadata, schedule=schedule_metadata, lr=lr)
        model.load_state_dict(state.model_state)
        if state.weight_state is not None:
            logvars.load_state_dict(state.weight_state)
        if state.optimizer_state is not None:
            opt.load_state_dict(state.optimizer_state)
            if schedule is None:
                for group in opt.param_groups:     # scheduled runs retain their decayed rate
                    group["lr"] = lr
        if engine.scheduler is not None:
            engine.scheduler.load_state_dict(state.scheduler_state)
        with error_context(path=str(resume_from)):
            _restore_rng(state.rng)
        start = state.step
        if steps <= start:  # nothing to do -> refuse rather than mis-save
            raise ValueError(f"resume target steps={steps} is not beyond the checkpoint step {start}; "
                             f"pass steps > {start} to continue training.")
        print(f"RESUMED from {resume_from} at step {start} (lr={lr}, short_boost={short_boost})", flush=True)
    elif init_from:                                # warm-start: copy the shape-compatible backbone
        from .transfer import load_source_state_dict, summarize, transfer_compatible_weights
        src_sd = load_source_state_dict(init_from, device=device, trust_pickle=init_trust_pickle)
        transferred, skipped = transfer_compatible_weights(model, src_sd)
        print(f"WARM-START from {init_from}\n{summarize(transferred, skipped)}", flush=True)

    _train_args = {"lr": lr, "batch_size": batch_size, "progresses": list(progresses),
                   "heavy_shm": heavy_shm, "short_boost": short_boost, "seed": seed, "steps": steps,
                   # `mutation_cap` is the run-level REQUEST; `effective_mutation_caps` is what each
                   # locus actually got (TCR -> 0). Without the latter the card misstates the training
                   # distribution: a TRA run reads `mutation_cap: null, heavy_shm: 0.25` yet saw no SHM.
                   "input_space": cfg.input_space, "label_space": cfg.label_space,
                   "source_mode": source_policy.mode, "mutation_cap": mutation_cap,
                   "augmentation": augmentation_metadata,
                   "schedule": schedule_metadata, "study": study.to_dict() if study is not None else None,
                   "effective_mutation_caps": effective_mutation_caps(dataconfigs, mutation_cap)}
    _dcs = dataconfigs
    source_checkpoint = resume_from or init_from
    if source_checkpoint:
        import hashlib
        with open(source_checkpoint, 'rb') as source:
            _train_args['source_checkpoint_sha256'] = hashlib.file_digest(source, 'sha256').hexdigest()

    def _checkpoint(path, step, **extra):
        save_checkpoint(path, cfg, model, logvars, step, opt, dataconfigs=_dcs,
                        train_args={**_train_args, **extra}, label_schema=schema, scheduler=engine.scheduler)

    observation = StudySession(study, model, _checkpoint) if study is not None else None

    # ModelXRay: a read-only training X-ray over a fixed held-out probe batch
    xray = probe_input = None
    if monitor_log:
        from ..xray import ModelXRay
        ev = list(itertools.islice(_build_stream(seed + 99991), batch_size))
        probe_input, probe_targets = data.validation_adapter(ev)

        def task_eval(m, inp, _tg=probe_targets):
            return tasks.metrics(m(inp), _tg)

        def task_losses(m, inp, _tg=probe_targets, _lv=logvars):
            return tasks.loss(m(inp), _tg, _lv)[1]      # per-task loss tensors

        xray = ModelXRay(model, lr=lr, log_path=monitor_log, deep_every=deep_every,
                         uncertainty=logvars, task_eval=task_eval, task_losses=task_losses,
                         shared_module="embedding")

    # fixed-seed, immutable validation batches for best-checkpoint selection
    val_set = []
    if val_every and val_every > 0:
        vstream = _build_stream(seed + 424242)         # a held-out seed, disjoint from training
        for _ in range(max(1, int(val_batches))):
            vr = list(itertools.islice(vstream, batch_size))
            if len(vr) < batch_size:
                break
            with error_context(stage="training_validation", validation_batch=len(val_set)):
                val_set.append(data.validation_adapter(vr))
    best_score = float("-inf")
    best_path = None

    completed = start                              # last successful update, including early exhaustion
    stream = _build_stream(seed + start, augment=True, start_step=start)
    batch_adapter = data.batch_adapter
    for iteration in training_iterations(engine, stream, batch_adapter, start_step=start,
                                         stop_step=steps, batch_size=batch_size):
        step = completed = iteration.step
        total, parts, grad_norm = iteration.result
        if on_iteration is not None:
            on_iteration(iteration)
        if observation is not None:
            observation.observe(iteration)
        if log_every and step % log_every == 0:
            line = (f"[{step}/{steps}] loss {total:.3f} gnorm {grad_norm:.2f} "
                    + " ".join(f"{k}={v:.2f}" for k, v in parts.items()))
            if xray is not None:                   # reads this step's grads (pre next zero_grad)
                rec = xray.observe(step, total, parts, probe_input=probe_input)
                line += "  " + xray.summary_line(rec)
            print(line, flush=True)
        if val_set and step % val_every == 0:      # validation + best-checkpoint selection
            vmetrics, vscore = engine.validate(val_set)
            print(f"[{step}/{steps}] VAL score {vscore:.4f}  "
                  + " ".join(f"{k}={v:.3f}" for k, v in vmetrics.items()), flush=True)
            if save_path and vscore == vscore and vscore > best_score:   # not NaN and improved
                best_score = vscore
                stem, ext = _stem(save_path), _ext(save_path)
                best_path = f"{stem}.best{ext}"
                _checkpoint(best_path, step, val_score=vscore)
        if save_path and save_every and step % save_every == 0:
            save_rotating(save_path, cfg, model, logvars, step, opt, dataconfigs=_dcs, train_args=_train_args,
                          label_schema=schema, scheduler=engine.scheduler)
    if save_path:                                  # save the ACTUAL completed step, not the requested one
        save_rotating(save_path, cfg, model, logvars, completed, opt, dataconfigs=_dcs, train_args=_train_args,
                      label_schema=schema, scheduler=engine.scheduler)
    if xray is not None:
        xray.close()
    return TrainingOutcome(model, completed, best_path, observation.checkpoints if observation else {},
                           tuple(observation.validations) if observation else ())
