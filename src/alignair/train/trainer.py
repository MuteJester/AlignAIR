"""Training loop for the AlignAIR model on GenAIRR record streams."""
from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, replace

from ..errors import ConfigurationError, error_context
from ..data.training import prepare_training_data
from ..core.setup import ModelSetup
from ..core.runtime import restore_model_state
from ..genairr.identity import recorded_genairr_source
from ..tasks import TrainingTasks, build_tasks
from .records.curriculum import INDEL_FRACTION
from ..genairr.catalog import dataconfig_name
from .records.streams import (SELECTED_FRACTION, CurriculumStreamPolicy, effective_mutation_caps,
                          selection_supported, training_records)
from .setup import build_training_engine
from .loop import training_iterations
from .checkpoints import (best_checkpoint_path, save_checkpoint, save_rotating, latest_checkpoint,
                          load_training_checkpoint, validate_continuation)
from .random_state import restore_rng_state as _restore_rng
from .schedules import CosineSchedule

# The id of how a run's curriculum records and targets are built, recorded as ``train_args["recipe"]``; it changes
# whenever the records or targets a run of the same options trains on change. "v4": every stream draws from its own
# reserved seed block, a selected fraction of each full-length component is drawn under productive_only(), the
# sequencing errors are at the chip's level (``SEQUENCING_ERROR_SCALE``), and the allele-set and mutation-rate targets
# are ``data.targets``'. The options recorded beside it (indel form and fraction, selected fraction) are its parameters.
RECIPE = 'v4'


@dataclass(frozen=True)
class TrainingOutcome:
    """This invocation's completed updates and best checkpoint, never inferred from old files."""
    model: object
    steps: int
    best_model_path: str | None


def _check_novel_alleles(dataconfigs, reference_set, spec) -> None:
    """ConfigurationError unless every dataconfig has an eligible V (``novel_support``) and every locus of the reference
    is of one and the same species (one species per model); prints what each epoch draws."""
    from .records.novel import novel_support
    species = {info.species for info in reference_set.loci}
    if len(species) != 1 or None in species:
        raise ConfigurationError(f"novel alleles train a model of one species; the reference's loci are of "
                                 f"{sorted(str(s) for s in species)}", context={"field": "novel_alleles"})
    for dataconfig in dataconfigs:
        genes = novel_support(dataconfig, spec)
        print(f"novel alleles: {dataconfig_name(dataconfig)}: {max(1, round(spec.gene_fraction * genes))} of {genes} "
              f"eligible V genes per epoch of {spec.epoch_records} records", flush=True)


def train(model, reference_set, dataconfigs, cfg, *, steps=2000, batch_size=32,
          lr=3e-4, progresses=(0.3, 0.6, 0.9), heavy_shm=0.25, short_boost=1, seed=0, init_seed=None, device="cpu",
          log_every=50, save_path=None, save_every=2000, resume_path=None,
          mutation_cap=None, init_from=None,
          grad_clip=None, val_every=0, val_batches=4,
          tasks: TrainingTasks | None = None, model_setup: ModelSetup | None = None, on_iteration=None,
          schedule=None, data_workers=0, record_source=None,
          indel_fraction=INDEL_FRACTION, indel_form='weighted', selected_fraction=SELECTED_FRACTION, flanks=None,
          novel_alleles=None, inverted_d=0.):
    """Train ``model`` on the curriculum stream of ``dataconfigs``.

    ``indel_form`` and ``indel_fraction`` select the curriculum's polymerase-indel counts
    (``records.streams.CurriculumStreamPolicy``): ``"weighted"``, whose weight of reads with an indel is
    ``indel_fraction`` ``(low, high)`` at progress 0 and 1, or ``"uniform"``, a uniform count. ``selected_fraction`` of
    each full-length component's records are drawn under GenAIRR's ``productive_only()``. The run records them in
    ``train_args`` with its ``recipe`` (``RECIPE``) and each component's selected share. When GenAIRR cannot draw
    productive records from one of ``dataconfigs`` (``records.streams.selection_supported``), the run draws no selected
    records, says so, and lists those references in ``train_args["selection_unavailable"]``.

    ``flanks`` (``data.flanks.FlankPolicy``) puts library flanks around every curriculum stream's reads
    (``CurriculumStreamPolicy.flanks``): training and validation; ``train_args["flanks"]`` records it.

    ``inverted_d`` inverts the D of that share of every curriculum stream's records
    (``CurriculumStreamPolicy.inverted_d``); a run with inverted Ds records it as ``train_args["inverted_d"]``.

    ``record_source`` (``records.streams.RecordSource``: ``(reference, parameters, seed) -> records``) replaces the
    GenAIRR provider of every curriculum stream of the run: training and validation. With
    ``data_workers`` > 0 it runs in the worker processes, so it must be picklable (a module-level function or a
    picklable object). None generates the records with GenAIRR.

    ``train_args`` also record the run's provenance: ``init_seed``, the seed the caller drew the model's initial
    weights from (``TrainingConfig.init_seed``; None: from ``seed``), ``grad_clip``, ``started_from`` (None for a
    run from its initial weights, ``"resume"`` or ``"init_from"``; either of these also records the
    ``source_checkpoint_sha256``) and the ``genairr_source_sha256`` of the GenAIRR it imports
    (``genairr_source_sha256``; None when a file of the package cannot be read).
    """
    tasks, dataconfigs = _preflight(model, cfg, reference_set, dataconfigs, tasks=tasks, steps=steps,
                                    batch_size=batch_size, lr=lr, progresses=progresses, heavy_shm=heavy_shm,
                                    short_boost=short_boost, grad_clip=grad_clip, data_workers=data_workers,
                                    record_source=record_source, on_iteration=on_iteration, init_from=init_from,
                                    resume_path=resume_path, schedule=schedule)
    setup = model_setup or ModelSetup(reference_set, cfg)
    if setup.reference is not reference_set or setup.config != cfg:
        raise ConfigurationError('training setup must bind this model configuration and reference')
    data = prepare_training_data(setup, device=device)
    schedule_metadata = schedule.to_dict() if schedule is not None else None
    requested_policy = CurriculumStreamPolicy(progresses, heavy_shm, short_boost, mutation_cap,
                                              mode='auto',
                                              indel_fraction=indel_fraction, indel_form=indel_form,
                                              selected_fraction=selected_fraction, flanks=flanks,
                                              novel_alleles=novel_alleles, inverted_d=inverted_d)
    if novel_alleles is not None:        # before any record is drawn: an eligible V, one species, GenAIRR's consent
        _check_novel_alleles(dataconfigs, reference_set, novel_alleles)
    source_policy, unselectable = _record_policy(requested_policy, dataconfigs)
    engine = build_training_engine(model, tasks=tasks, device=device, lr=lr, grad_clip=grad_clip,
                                   scheduler_factory=schedule.build if schedule is not None else None)
    # the order that matters: the RNG restore (a resume), then the validation batches, then the training stream
    start, resume_from = _restore_or_warm_start(model, engine, reference_set, cfg, resume_path=resume_path,
                                                init_from=init_from, steps=steps, lr=lr, schedule=schedule,
                                                schedule_metadata=schedule_metadata, short_boost=short_boost,
                                                device=device)
    train_args = _train_args(
        dataconfigs, requested_policy, source_policy, unselectable, lr=lr, batch_size=batch_size,
        progresses=progresses, heavy_shm=heavy_shm, short_boost=short_boost, seed=seed, steps=steps,
        mutation_cap=mutation_cap, flanks=flanks, schedule_metadata=schedule_metadata, data_workers=data_workers,
        init_seed=init_seed, grad_clip=grad_clip, started_from="resume" if resume_from else
        "init_from" if init_from else None, novel_alleles=novel_alleles, source_checkpoint=resume_from or init_from)
    val_set = _validation_batches(dataconfigs, source_policy, data, seed=seed, record_source=record_source,
                                  batch_size=batch_size, val_every=val_every, val_batches=val_batches)
    stream, batch_adapter, workers = _training_source(dataconfigs, source_policy, setup, data, seed=seed, start=start,
                                                      batch_size=batch_size, data_workers=data_workers,
                                                      device=device, record_source=record_source)
    best_score, best_path = float("-inf"), None
    completed = start                              # last successful update, including early exhaustion
    try:
        for iteration in training_iterations(engine, stream, batch_adapter, start_step=start,
                                             stop_step=steps, batch_size=batch_size):
            step = completed = iteration.step
            if on_iteration is not None:
                on_iteration(iteration)
            if log_every and step % log_every == 0:
                _log_iteration(iteration, steps)
            if val_set and step % val_every == 0:  # validation + best-checkpoint selection
                vscore = _validate(engine, val_set, step, steps)
                if save_path and vscore == vscore and vscore > best_score:   # not NaN and improved
                    best_score, best_path = vscore, best_checkpoint_path(save_path)
                    save_checkpoint(best_path, model, reference_set, step, engine.optimizer, dataconfigs=dataconfigs,
                                    train_args={**train_args, "val_score": vscore}, scheduler=engine.scheduler)
            if save_path and save_every and step % save_every == 0:
                save_rotating(save_path, model, reference_set, step, engine.optimizer, dataconfigs=dataconfigs,
                              train_args=train_args, scheduler=engine.scheduler)
    finally:
        if workers is not None:                    # finished, failed or interrupted: no worker outlives the run
            workers.close()
    if save_path:                                  # save the ACTUAL completed step, not the requested one
        save_rotating(save_path, model, reference_set, completed, engine.optimizer, dataconfigs=dataconfigs,
                      train_args=train_args, scheduler=engine.scheduler)
    return TrainingOutcome(model, completed, best_path)


def _preflight(model, cfg, reference_set, dataconfigs, *, tasks, steps, batch_size, lr, progresses, heavy_shm,
               short_boost, grad_clip, data_workers, record_source, on_iteration, init_from, resume_path, schedule):
    """``(tasks, dataconfigs)``: the run's tasks (the model's own unless given) and its dataconfigs as a list, after
    refusing a bad request up front rather than after a long run (ConfigurationError, TrainingConfigError)."""
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
    validate_training_request(steps=steps, batch_size=batch_size, lr=lr,
                              max_seq_length=cfg.max_seq_length, reference=reference_set,
                              progresses=progresses, heavy_shm=heavy_shm, short_boost=short_boost,
                              grad_clip=grad_clip)
    if on_iteration is not None and not callable(on_iteration):
        raise ConfigurationError('on_iteration must be callable or None')
    if init_from and resume_path:
        raise ConfigurationError('choose resume_path or init_from')
    if schedule is not None:
        if not isinstance(schedule, CosineSchedule):
            raise ConfigurationError('schedule must be a CosineSchedule or None')
        schedule.validate(lr)
    return tasks, dataconfigs


def _record_policy(requested, dataconfigs) -> tuple[CurriculumStreamPolicy, list[str]]:
    """``(policy, unselectable)``: the policy the run draws its records with, and the dataconfigs GenAIRR cannot draw
    productive records from (an incompletely annotated custom reference). With any, the whole run draws no selected
    records, says so, and records them in train_args."""
    unselectable = ([dataconfig_name(dataconfig) for dataconfig in dataconfigs if not selection_supported(dataconfig)]
                    if requested.selected_fraction else [])
    if unselectable:
        print(f"selection unavailable for {', '.join(unselectable)}: GenAIRR cannot draw productive records from "
              "it, so the run draws no selected records", flush=True)
    return (replace(requested, selected_fraction=0) if unselectable else requested), unselectable


def _restore_or_warm_start(model, engine, reference_set, cfg, *, resume_path, init_from, steps, lr, schedule,
                           schedule_metadata, short_boost, device) -> tuple[int, str | None]:
    """``(start, resume_from)``: a resume restores the latest checkpoint at ``resume_path`` (weights, optimizer,
    scheduler and the process RNG) and starts at its step; a warm start (``init_from``) copies the shape-compatible
    weights and starts at 0."""
    resume_from = latest_checkpoint(resume_path) if resume_path else None
    if resume_path and resume_from is None:
        raise ConfigurationError('resume checkpoint does not exist', context={'path': str(resume_path)})
    if resume_from:
        state = load_training_checkpoint(resume_from, device=device)
        validate_continuation(state, cfg, reference_set, schedule=schedule_metadata, lr=lr)
        restore_model_state(model, state.model_state, source=resume_from)
        if state.optimizer_state is not None:
            engine.optimizer.load_state_dict(state.optimizer_state)
            if schedule is None:
                for group in engine.optimizer.param_groups:     # scheduled runs retain their decayed rate
                    group["lr"] = lr
        if engine.scheduler is not None:
            engine.scheduler.load_state_dict(state.scheduler_state)
        with error_context(path=str(resume_from)):
            _restore_rng(state.rng)
        if steps <= state.step:  # nothing to do -> refuse rather than mis-save
            raise ConfigurationError(f"resume target steps={steps} is not beyond the checkpoint step {state.step}; "
                                     f"pass steps > {state.step} to continue training.", context={"field": "steps"})
        print(f"RESUMED from {resume_from} at step {state.step} (lr={lr}, short_boost={short_boost})", flush=True)
        return state.step, resume_from
    if init_from:                                  # warm-start: copy the shape-compatible backbone
        from .transfer import load_source_state_dict, summarize, transfer_compatible_weights
        src_sd = load_source_state_dict(init_from, device=device)
        transferred, skipped = transfer_compatible_weights(model, src_sd)
        print(f"WARM-START from {init_from}\n{summarize(transferred, skipped)}", flush=True)
    return 0, None


def _train_args(dataconfigs, requested_policy, source_policy, unselectable, *, lr, batch_size, progresses, heavy_shm,
                short_boost, seed, steps, mutation_cap, flanks, schedule_metadata, data_workers, init_seed, grad_clip,
                started_from, novel_alleles, source_checkpoint) -> dict:
    """The run's ``train_args``: its request, its recipe and its provenance (see ``train``)."""
    args = {"lr": lr, "batch_size": batch_size, "progresses": list(progresses),
            "heavy_shm": heavy_shm, "short_boost": short_boost, "seed": seed, "steps": steps,
            # `mutation_cap` is the run-level REQUEST; `effective_mutation_caps` is what each
            # locus actually got (TCR -> 0). Without the latter the card misstates the training
            # distribution: a TRA run reads `mutation_cap: null, heavy_shm: 0.25` yet saw no SHM.
            "source_mode": source_policy.mode, "mutation_cap": mutation_cap,
            "recipe": RECIPE,
            "indel_form": source_policy.indel_form, "indel_fraction": list(source_policy.indel_fraction),
            "selected_fraction": requested_policy.selected_fraction,
            "selected_shares": list(source_policy.selected_shares(force_mixture=len(dataconfigs) > 1)),
            "selection_unavailable": unselectable,
            "flanks": flanks.to_dict() if flanks is not None else None,
            "schedule": schedule_metadata,
            "effective_mutation_caps": effective_mutation_caps(dataconfigs, mutation_cap),
            # provenance only: worker processes reproduce the serial batches exactly
            "data_workers": data_workers,
            "init_seed": init_seed, "grad_clip": grad_clip,
            "started_from": started_from,
            "genairr_source_sha256": recorded_genairr_source()}
    if novel_alleles is not None:
        args["novel_alleles"] = novel_alleles.to_dict()
    if requested_policy.inverted_d:
        args["inverted_d"] = requested_policy.inverted_d
    if source_checkpoint:
        with open(source_checkpoint, 'rb') as source:
            args['source_checkpoint_sha256'] = hashlib.file_digest(source, 'sha256').hexdigest()
    return args


def _records(dataconfigs, policy, purpose: str, *, seed, record_source, start_step: int = 0):
    """A curriculum record stream: each draws its own seed blocks (training resumes from (seed, step);
    validation's are held out, disjoint from training's)."""
    return training_records(dataconfigs, policy=policy, seed=seed, purpose=purpose,
                            start_step=start_step, record_source=record_source)


def _validation_batches(dataconfigs, policy, data, *, seed, record_source, batch_size, val_every, val_batches) -> list:
    """The fixed-seed, immutable validation batches best-checkpoint selection scores (none without ``val_every``);
    fewer when the stream runs out."""
    val_set = []
    if val_every and val_every > 0:
        vstream = _records(dataconfigs, policy, 'validation', seed=seed, record_source=record_source)
        for _ in range(max(1, int(val_batches))):
            vr = list(itertools.islice(vstream, batch_size))
            if len(vr) < batch_size:
                break
            with error_context(stage="training_validation", validation_batch=len(val_set)):
                val_set.append(data.validation_adapter(vr))
    return val_set


def _training_source(dataconfigs, policy, setup, data, *, seed, start, batch_size, data_workers, device,
                     record_source):
    """``(records, batch_adapter, workers)``: the training stream and its adapter, built serially or, with
    ``data_workers``, the same batches in worker processes (``workers``, to close; None when serial)."""
    if not data_workers:
        return _records(dataconfigs, policy, 'training', seed=seed, record_source=record_source,
                        start_step=start), data.batch_adapter, None
    from .assembly import TrainingStream, parallel_training_data
    from .records.streams import stream_records
    workers = parallel_training_data(
        TrainingStream(tuple(dataconfigs), policy, seed, prepare_training_data(setup, device='cpu'), start_step=start),
        batch_size=batch_size, workers=data_workers, device=device,
        provider=stream_records if record_source is None else record_source)
    return workers.records, workers.batch_adapter, workers


def _log_iteration(iteration, steps: int) -> None:
    total, parts, grad_norm = iteration.result
    print(f"[{iteration.step}/{steps}] loss {total:.3f} gnorm {grad_norm:.2f} "
          + " ".join(f"{k}={v:.2f}" for k, v in parts.items()), flush=True)


def _validate(engine, val_set: list, step: int, steps: int) -> float:
    """The validation score of the model now (printed with its metrics)."""
    vmetrics, vscore = engine.validate(val_set)
    print(f"[{step}/{steps}] VAL score {vscore:.4f}  "
          + " ".join(f"{k}={v:.3f}" for k, v in vmetrics.items()), flush=True)
    return vscore
