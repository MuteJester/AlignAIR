"""Training loop for the AlignAIR model on GenAIRR record streams."""
from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..core.runtime import restore_model_state
from ..core.setup import ModelSetup
from ..data.training import prepare_training_data
from ..errors import ConfigurationError, error_context
from ..genairr.catalog import dataconfig_name
from ..genairr.identity import recorded_genairr_source
from ..tasks import TrainingTasks, build_tasks
from .checkpoints import (
    best_checkpoint_path,
    latest_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
    save_rotating,
    validate_continuation,
)
from .loop import training_iterations
from .random_state import restore_rng_state as _restore_rng
from .records.streams import (
    INDEL_FORM,
    CurriculumStreamPolicy,
    effective_mutation_caps,
    selection_supported,
    training_records,
)
from .setup import build_training_engine

if TYPE_CHECKING:
    from .config import TrainingConfig

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


def _novel_policy(policy: CurriculumStreamPolicy, dataconfigs, reference_set) -> tuple[CurriculumStreamPolicy,
                                                                                       str | None]:
    """``(policy, unavailable)``: the policy the run draws its records with, and why it draws no novel alleles though
    ``policy`` asks for them (None when it draws them, or none were asked for). Novel alleles cannot be drawn for a
    reference of several species or of none, or for a dataconfig with no eligible V (``_check_novel_alleles``); the
    run then trains on the reference's alleles alone, as a run GenAIRR cannot select records for trains unselected."""
    if policy.novel_alleles is None:
        return policy, None
    try:
        _check_novel_alleles(dataconfigs, reference_set, policy.novel_alleles)
    except ConfigurationError as error:
        if error.context.get("field") != "novel_alleles":
            raise
        print(f"novel alleles unavailable: {error.message}; the run trains on the reference's alleles alone",
              flush=True)
        return replace(policy, novel_alleles=None), error.message
    return policy, None


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


def _train(setup: ModelSetup, model, sources, config: TrainingConfig, policy: CurriculumStreamPolicy, *,
           device: str = "cpu", save_path: str | None = None, on_iteration=None, record_source=None,
           log_every: int = 50, save_every: int = 2000, val_batches: int = 4) -> TrainingOutcome:
    """Train ``model``, built from ``setup``, for ``config`` on ``policy``'s curriculum records of ``sources`` (the
    run's resolved dataconfigs); ``run_training`` is the public way in, and ``config.record_policy()`` its policy.

    ``policy.indel_fraction`` weights the curriculum's polymerase-indel counts
    (``records.streams.CurriculumStreamPolicy``), and ``policy.selected_fraction`` of each full-length component's
    records are drawn under GenAIRR's ``productive_only()``. The run records them in ``train_args`` with its ``recipe``
    (``RECIPE``) and each component's selected share. When GenAIRR cannot draw productive records from one of
    ``sources`` (``records.streams.selection_supported``), the run draws no selected records, says so, and lists those
    references in ``train_args["selection_unavailable"]``; when novel alleles cannot be drawn for them
    (``_novel_policy``), it draws none, says so, and records why in ``train_args["novel_alleles_unavailable"]``.
    ``train_args["flanks"]`` records the policy's flanks, and a run with inverted Ds records
    ``train_args["inverted_d"]``.

    ``record_source`` (``records.streams.RecordSource``: ``(reference, parameters, seed) -> records``) replaces the
    GenAIRR provider of every curriculum stream of the run: training and validation. With
    ``config.data_workers`` > 0 it runs in the worker processes, so it must be picklable (a module-level function or a
    picklable object). None generates the records with GenAIRR.

    ``train_args`` also record the run's provenance: ``init_seed``, the seed the caller drew the model's initial
    weights from (``TrainingConfig.init_seed``; None: from ``seed``), ``grad_clip``, ``started_from`` (None for a
    run from its initial weights, ``"resume"`` or ``"init_from"``; either of these also records the
    ``source_checkpoint_sha256``) and the ``genairr_source_sha256`` of the GenAIRR it imports
    (``genairr_source_sha256``; None when a file of the package cannot be read).
    """
    tasks, sources = _preflight(setup, model, sources, record_source=record_source, on_iteration=on_iteration)
    reference_set, steps, batch_size, val_every = setup.reference, config.steps, config.batch_size, config.val_every
    data = prepare_training_data(setup, device=device)
    schedule = config.schedule
    schedule_metadata = schedule.to_dict() if schedule is not None else None
    # before any record is drawn: an eligible V, one species, GenAIRR's consent
    novel_policy, novel_unavailable = _novel_policy(policy, sources, reference_set)
    source_policy, unselectable = _record_policy(novel_policy, sources)
    engine = build_training_engine(model, tasks=tasks, device=device, lr=config.lr, grad_clip=config.grad_clip,
                                   schedule=schedule)
    # the order that matters: the RNG restore (a resume), then the validation batches, then the training stream
    start, resume_from = _restore_or_warm_start(model, engine, setup, config, schedule_metadata, device=device)
    train_args = _train_args(sources, config, policy, source_policy, unselectable, novel_unavailable, schedule_metadata,
                             started_from="resume" if resume_from else "init_from" if config.init_from else None,
                             source_checkpoint=resume_from or config.init_from)
    val_set = _validation_batches(sources, source_policy, data, config, record_source=record_source,
                                  val_batches=val_batches)
    stream, batch_adapter, workers = _training_source(sources, source_policy, setup, data, config, start=start,
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
                if save_path and not math.isnan(vscore) and vscore > best_score:
                    best_score, best_path = vscore, best_checkpoint_path(save_path)
                    save_checkpoint(best_path, model, reference_set, step, engine.optimizer, dataconfigs=sources,
                                    train_args={**train_args, "val_score": vscore}, scheduler=engine.scheduler)
            if save_path and save_every and step % save_every == 0:
                save_rotating(save_path, model, reference_set, step, engine.optimizer, dataconfigs=sources,
                              train_args=train_args, scheduler=engine.scheduler)
    finally:
        if workers is not None:                    # finished, failed or interrupted: no worker outlives the run
            workers.close()
    if save_path:                                  # save the ACTUAL completed step, not the requested one
        save_rotating(save_path, model, reference_set, completed, engine.optimizer, dataconfigs=sources,
                      train_args=train_args, scheduler=engine.scheduler)
    return TrainingOutcome(model, completed, best_path)


def _preflight(setup: ModelSetup, model, sources, *, record_source, on_iteration) -> tuple[TrainingTasks, list]:
    """``(tasks, sources)``: the model's tasks and the run's dataconfigs as a list, after refusing a model that is not
    the setup's and a record source or callback that cannot be called (ConfigurationError). ``TrainingConfig``
    validated the rest at construction."""
    model_config = getattr(model, 'cfg', setup.config)
    if model_config != setup.config:
        raise ConfigurationError('training configuration differs from the constructed model',
                                 context={'field': 'config'},
                                 hint='Construct a new model with the intended configuration before training.')
    if record_source is not None and not callable(record_source):
        raise ConfigurationError('record_source must be callable or None', context={'field': 'record_source'})
    if on_iteration is not None and not callable(on_iteration):
        raise ConfigurationError('on_iteration must be callable or None')
    tasks = build_tasks(model_config)
    return tasks, list(sources) if isinstance(sources, (list, tuple)) else [sources]


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


def _restore_or_warm_start(model, engine, setup: ModelSetup, config: TrainingConfig, schedule_metadata, *,
                           device) -> tuple[int, str | None]:
    """``(start, resume_from)``: a resume restores the latest checkpoint at ``config.resume_path`` (weights, optimizer,
    scheduler and the process RNG) and starts at its step; a warm start (``config.init_from``) copies the
    shape-compatible weights and starts at 0."""
    resume_path, init_from, steps, lr = config.resume_path, config.init_from, config.steps, config.lr
    schedule, short_boost, reference_set, cfg = config.schedule, config.short_boost, setup.reference, setup.config
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


def _train_args(sources, config: TrainingConfig, requested_policy, source_policy, unselectable, novel_unavailable,
                schedule_metadata, *, started_from, source_checkpoint) -> dict:
    """The run's ``train_args``: its request, its recipe and its provenance (see ``_train``)."""
    requested = requested_policy
    args = {"lr": config.lr, "batch_size": config.batch_size, "progresses": list(requested.progresses),
            "heavy_shm": requested.heavy_shm, "short_boost": requested.short_boost, "seed": config.seed,
            "steps": config.steps,
            # source_mode, mutation_cap and indel_form are the values every run has trained with (the stream policy no
            # longer has the fields), kept so train_args compare with every earlier checkpoint's;
            # `effective_mutation_caps` is what each locus got (TCR -> 0).
            "source_mode": "auto", "mutation_cap": None,
            "recipe": RECIPE,
            "indel_form": INDEL_FORM, "indel_fraction": list(source_policy.indel_fraction),
            "selected_fraction": requested.selected_fraction,
            "selected_shares": list(source_policy.selected_shares(force_mixture=len(sources) > 1)),
            "selection_unavailable": unselectable,
            "flanks": requested.flanks.to_dict() if requested.flanks is not None else None,
            "schedule": schedule_metadata,
            "effective_mutation_caps": effective_mutation_caps(sources),
            # provenance only: worker processes reproduce the serial batches exactly
            "data_workers": config.data_workers,
            "init_seed": config.init_seed, "grad_clip": config.grad_clip,
            "started_from": started_from,
            "genairr_source_sha256": recorded_genairr_source()}
    if source_policy.novel_alleles is not None:              # what the run drew, and why not when it was asked for
        args["novel_alleles"] = source_policy.novel_alleles.to_dict()
    if novel_unavailable is not None:
        args["novel_alleles_unavailable"] = novel_unavailable
    if requested.inverted_d:
        args["inverted_d"] = requested.inverted_d
    if source_checkpoint:
        with open(source_checkpoint, 'rb') as source:
            args['source_checkpoint_sha256'] = hashlib.file_digest(source, 'sha256').hexdigest()
    return args


def _records(dataconfigs, policy, purpose: str, *, seed, record_source, start_step: int = 0):
    """A curriculum record stream: each draws its own seed blocks (training resumes from (seed, step);
    validation's are held out, disjoint from training's)."""
    return training_records(dataconfigs, policy=policy, seed=seed, purpose=purpose,
                            start_step=start_step, record_source=record_source)


def _validation_batches(dataconfigs, policy, data, config: TrainingConfig, *, record_source, val_batches) -> list:
    """The fixed-seed, immutable validation batches best-checkpoint selection scores (none without ``val_every``);
    fewer when the stream runs out."""
    seed, batch_size, val_every = config.seed, config.batch_size, config.val_every
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


def _training_source(dataconfigs, policy, setup, data, config: TrainingConfig, *, start, device, record_source):
    """``(records, batch_adapter, workers)``: the training stream and its adapter, built serially or, with
    ``config.data_workers``, the same batches in worker processes (``workers``, to close; None when serial)."""
    seed, batch_size, data_workers = config.seed, config.batch_size, config.data_workers
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
