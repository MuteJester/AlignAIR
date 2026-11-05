"""Training loop for the AlignAIR model on GenAIRR record streams.

``build_batch`` is a compatibility import of the shared data adapter.
"""
from __future__ import annotations

import functools
import hashlib
import itertools
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..errors import ConfigurationError, error_context
from ..data.adapters import build_nucleotide_batch as build_batch
from ..data.training import prepare_training_data
from ..core.setup import ModelSetup
from ..core.runtime import restore_model_state
from ..tasks import TrainingTasks, build_tasks
from .gym.curriculum.base import INDEL_FRACTION
from .gym.streams import (SELECTED_FRACTION, CurriculumStreamPolicy, dataconfig_name, effective_mutation_caps,
                          selection_supported, training_records)
from .engine import validate
from .setup import build_training_engine
from .loop import training_iterations
from .checkpoints import (save_checkpoint, save_rotating, latest_checkpoint, load_training_checkpoint,
                          validate_continuation, _stem, _ext)
from .random_state import restore_rng_state as _restore_rng
from .schedules import CosineSchedule
from .study import TrainingStudy, StudySession

# How a run's curriculum records are built, recorded as ``train_args["recipe"]``: "v3" draws every stream from its
# own reserved seed block and a selected fraction of each full-length component under productive_only(). The options
# recorded beside it (indel form and fraction, selected fraction) are the recipe's parameters.
RECIPE = 'v3'


@functools.lru_cache(maxsize=None)
def _tree_sha256(directory: str) -> str:
    """SHA-256 over every file under ``directory`` but ``__pycache__`` and ``.pyc``: in the sorted order of their
    POSIX paths relative to it, each path, its byte length and its bytes. Computed once per process."""
    root = Path(directory)
    names = sorted(path.relative_to(root).as_posix() for path in root.rglob('*')
                   if path.is_file() and '__pycache__' not in path.relative_to(root).parts and path.suffix != '.pyc')
    digest = hashlib.sha256()
    for name in names:
        data = (root / name).read_bytes()
        digest.update(f'{name}\0{len(data)}\0'.encode())
        digest.update(data)
    return digest.hexdigest()


def genairr_source_sha256() -> str:
    """The content digest of the GenAIRR package this process imports (its sources, compiled engine and data), as
    ``train_args["genairr_source_sha256"]`` records it: the method of the assay's GenAIRR provenance
    (``tree_digest`` of the package directory), so the two compare. About 0.5 s, once per process."""
    import GenAIRR
    return _tree_sha256(str(Path(GenAIRR.__file__).resolve().parent))


def _recorded_genairr_source() -> str | None:
    """``genairr_source_sha256()``, or None, said on one line, when a file of the package cannot be read: the digest
    is provenance, and never fails a run."""
    try:
        return genairr_source_sha256()
    except OSError as error:
        print(f"GenAIRR source digest not recorded: {error}", flush=True)
        return None


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


def train(model, reference_set, dataconfigs, cfg, *, steps=2000, batch_size=32,
          lr=3e-4, progresses=(0.3, 0.6, 0.9), heavy_shm=0.25, short_boost=1, seed=0, init_seed=None, device="cpu",
          log_every=50, save_path=None, save_every=2000, resume_path=None,
          mutation_cap=None, init_from=None, init_trust_pickle=False,
          grad_clip=None, val_every=0, val_batches=4, resume_trust_pickle=False,
          tasks: TrainingTasks | None = None, model_setup: ModelSetup | None = None, on_iteration=None,
          schedule=None, study=None, start_step=0, data_workers=0, record_source=None,
          indel_fraction=INDEL_FRACTION, indel_form='weighted', selected_fraction=SELECTED_FRACTION, flanks=None):
    """Train ``model`` on the curriculum stream of ``dataconfigs``.

    ``indel_form`` and ``indel_fraction`` select the curriculum's polymerase-indel counts
    (``gym.streams.CurriculumStreamPolicy``): ``"weighted"``, whose weight of reads with an indel is
    ``indel_fraction`` ``(low, high)`` at progress 0 and 1, or the historical ``"uniform"`` form of the 1.0.0
    recipe. ``selected_fraction`` of each full-length component's records are drawn under GenAIRR's
    ``productive_only()``. The run records them in ``train_args`` with its ``recipe`` (``RECIPE``) and each
    component's selected share. When GenAIRR cannot draw productive records from one of ``dataconfigs``
    (``gym.streams.selection_supported``), the run draws no selected records, says so, and lists those references
    in ``train_args["selection_unavailable"]``.

    ``flanks`` (``data.flanks.FlankPolicy``) puts library flanks around every curriculum stream's reads
    (``CurriculumStreamPolicy.flanks``): training and validation; ``train_args["flanks"]`` records it.

    ``record_source`` (``gym.streams.RecordSource``: ``(reference, parameters, seed) -> records``) replaces the
    GenAIRR provider of every curriculum stream of the run: training and validation. With
    ``data_workers`` > 0 it runs in the worker processes, so it must be picklable (a module-level function or a
    picklable object). None generates the records with GenAIRR.

    ``train_args`` also record the run's provenance: ``init_seed``, the seed the caller drew the model's initial
    weights from (``TrainingConfig.init_seed``; None: from ``seed``), ``grad_clip``, ``started_from`` (None for a
    run from its initial weights, ``"resume"`` or ``"init_from"``; either of these also records the
    ``source_checkpoint_sha256``) and the ``genairr_source_sha256`` of the GenAIRR it imports
    (``genairr_source_sha256``; None when a file of the package cannot be read).
    """
    from .guards import validate_data_workers, validate_training_request
    model_config = getattr(model, 'cfg', cfg)
    if model_config != cfg:
        raise ConfigurationError('training configuration differs from the constructed model',
                                 context={'field': 'config'},
                                 hint='Construct a new model with the intended configuration before training.')
    validate_data_workers(data_workers)
    if record_source is not None and not callable(record_source):
        raise ConfigurationError('record_source must be callable or None', context={'field': 'record_source'})
    tasks = build_tasks(model_config) if tasks is None else tasks
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

    setup = model_setup or ModelSetup(reference_set, cfg, cfg.input_space, cfg.label_space)
    if setup.reference is not reference_set or setup.config != cfg:
        raise ConfigurationError('training setup must bind this model configuration and reference')
    data = prepare_training_data(setup, device=device)
    schedule_metadata = schedule.to_dict() if schedule is not None else None
    schema = setup.label_schema()
    requested_policy = CurriculumStreamPolicy(progresses, heavy_shm, short_boost, mutation_cap,
                                              mode='auto',
                                              indel_fraction=indel_fraction, indel_form=indel_form,
                                              selected_fraction=selected_fraction, flanks=flanks)
    # A reference GenAIRR cannot draw productive records from (an incompletely annotated custom reference) trains
    # unselected: the whole run then draws no selected records, says so, and records it in train_args.
    unselectable = ([dataconfig_name(dataconfig) for dataconfig in dataconfigs if not selection_supported(dataconfig)]
                    if requested_policy.selected_fraction else [])
    source_policy = replace(requested_policy, selected_fraction=0) if unselectable else requested_policy
    if unselectable:
        print(f"selection unavailable for {', '.join(unselectable)}: GenAIRR cannot draw productive records from "
              "it, so the run draws no selected records", flush=True)

    def _build_stream(purpose, *, start_step=0):
        # every stream draws its own seed blocks: training (a resume from (seed, step)) and validation
        return training_records(dataconfigs, policy=source_policy, seed=seed, purpose=purpose,
                                start_step=start_step, record_source=record_source)

    engine = build_training_engine(model, tasks=tasks, device=device, lr=lr, grad_clip=grad_clip,
                                   scheduler_factory=schedule.build if schedule is not None else None)
    opt = engine.optimizer

    start = start_step
    resume_from = latest_checkpoint(resume_path) if resume_path else None
    if resume_path and resume_from is None:
        raise ConfigurationError('resume checkpoint does not exist', context={'path': str(resume_path)})
    if resume_from:
        state = load_training_checkpoint(resume_from, device=device, trust_pickle=resume_trust_pickle)
        validate_continuation(state, cfg, schema, schedule=schedule_metadata, lr=lr)
        restore_model_state(model, state.model_state, source=resume_from)
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
                   "recipe": RECIPE,
                   "indel_form": source_policy.indel_form, "indel_fraction": list(source_policy.indel_fraction),
                   "selected_fraction": requested_policy.selected_fraction,
                   "selected_shares": list(source_policy.selected_shares(force_mixture=len(dataconfigs) > 1)),
                   "selection_unavailable": unselectable,
                   "flanks": flanks.to_dict() if flanks is not None else None,
                   "schedule": schedule_metadata, "study": study.to_dict() if study is not None else None,
                   "effective_mutation_caps": effective_mutation_caps(dataconfigs, mutation_cap),
                   # provenance only: worker processes reproduce the serial batches exactly
                   "data_workers": data_workers,
                   "init_seed": init_seed, "grad_clip": grad_clip,
                   "started_from": "resume" if resume_from else "init_from" if init_from else None,
                   "genairr_source_sha256": _recorded_genairr_source()}
    _dcs = dataconfigs
    source_checkpoint = resume_from or init_from
    if source_checkpoint:
        with open(source_checkpoint, 'rb') as source:
            _train_args['source_checkpoint_sha256'] = hashlib.file_digest(source, 'sha256').hexdigest()

    def _checkpoint(path, step, **extra):
        save_checkpoint(path, cfg, model, step, opt, dataconfigs=_dcs,
                        train_args={**_train_args, **extra}, label_schema=schema, scheduler=engine.scheduler)

    observation = StudySession(study, model, _checkpoint) if study is not None else None

    # fixed-seed, immutable validation batches for best-checkpoint selection
    val_set = []
    if val_every and val_every > 0:
        vstream = _build_stream('validation')          # held-out seed blocks, disjoint from training
        for _ in range(max(1, int(val_batches))):
            vr = list(itertools.islice(vstream, batch_size))
            if len(vr) < batch_size:
                break
            with error_context(stage="training_validation", validation_batch=len(val_set)):
                val_set.append(data.validation_adapter(vr))
    best_score = float("-inf")
    best_path = None

    completed = start                              # last successful update, including early exhaustion
    workers = None
    if data_workers:                               # the same batches, generated in worker processes
        from .assembly import TrainingStream, parallel_training_data
        from .gym.streams import stream_records
        workers = parallel_training_data(
            TrainingStream(tuple(dataconfigs), source_policy, seed,
                           prepare_training_data(setup, device='cpu'), start_step=start),
            batch_size=batch_size, workers=data_workers, device=device,
            provider=stream_records if record_source is None else record_source)
        stream, batch_adapter = workers.records, workers.batch_adapter
    else:
        stream = _build_stream('training', start_step=start)
        batch_adapter = data.batch_adapter
    try:
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
                print(line, flush=True)
            if val_set and step % val_every == 0:  # validation + best-checkpoint selection
                vmetrics, vscore = engine.validate(val_set)
                print(f"[{step}/{steps}] VAL score {vscore:.4f}  "
                      + " ".join(f"{k}={v:.3f}" for k, v in vmetrics.items()), flush=True)
                if save_path and vscore == vscore and vscore > best_score:   # not NaN and improved
                    best_score = vscore
                    stem, ext = _stem(save_path), _ext(save_path)
                    best_path = f"{stem}.best{ext}"
                    _checkpoint(best_path, step, val_score=vscore)
            if save_path and save_every and step % save_every == 0:
                save_rotating(save_path, cfg, model, step, opt, dataconfigs=_dcs, train_args=_train_args,
                              label_schema=schema, scheduler=engine.scheduler)
    finally:
        if workers is not None:                    # finished, failed or interrupted: no worker outlives the run
            workers.close()
    if save_path:                                  # save the ACTUAL completed step, not the requested one
        save_rotating(save_path, cfg, model, completed, opt, dataconfigs=_dcs, train_args=_train_args,
                      label_schema=schema, scheduler=engine.scheduler)
    return TrainingOutcome(model, completed, best_path, observation.checkpoints if observation else {},
                           tuple(observation.validations) if observation else ())
