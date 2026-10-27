"""Training loop for the faithful AlignAIR model on the GenAIRR gym stream.

``build_batch`` is a compatibility import of the shared data adapter. ``train_step`` runs one
optimizer step and applies the TF weight *constraints* (analysis-head kernel clamps + Kendall
log-var clamps) after ``optimizer.step()``.
"""
from __future__ import annotations

import itertools

import torch

from ..errors import error_context
from ..data.adapters import NucleotideBatchAdapter, build_nucleotide_batch as build_batch
from ..tasks import TrainingTasks, build_tasks
from .gym import build_experiment
from .engine import TrainingEngine, train_step, validate
from .loop import training_iterations
from .checkpoints import save_checkpoint, save_rotating, latest_checkpoint, load_training_checkpoint, _stem, _ext
from .random_state import restore_rng_state as _restore_rng


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
    if mutation_cap is not None:                    # loci without SHM (TCR): cap the S5F mutation rate
        for s in specs:
            s["mutation_rate"] = min(s["mutation_rate"], mutation_cap)
    return specs


def _is_tcr(dataconfig) -> bool:
    """TCR loci do not undergo somatic hypermutation (T-cells lack AID), and GenAIRR refuses
    ``mutate()`` on them."""
    return "TCR" in str(getattr(dataconfig.metadata, "chain_type", "")).upper()


def _cap_for(dataconfig, mutation_cap):
    """The SHM cap for ONE locus's stream. A TCR locus is always capped to 0 regardless of the
    run-level cap; an IG locus keeps the run-level cap (normally ``None`` = uncapped). Deriving this
    per dataconfig is what lets a MIXED IG+TCR run keep full SHM on its IG streams — a single global
    cap (``0.0`` if any locus is TCR) would wrongly disable SHM for IGH."""
    return 0.0 if _is_tcr(dataconfig) else mutation_cap


def _dc_name(dataconfig) -> str:
    """A stable, human-meaningful key for one dataconfig: its canonical GenAIRR name (e.g.
    ``HUMAN_TCRA_IMGT``) for a built-in, else its own name, else its chain type (custom references)."""
    try:
        import GenAIRR.data as gd
        # `vars()` = already-materialized module attributes only, so this never triggers a lazy build
        # of every shipped dataconfig (a dir()+getattr scan does, and costs ~0.5s).
        for k, v in vars(gd).items():
            if v is dataconfig and not k.startswith("_"):
                return k
    except Exception:                                  # noqa: BLE001 - naming must never break training
        pass
    name = getattr(dataconfig, "name", None)
    if name and str(name).strip().lower() not in ("none", "unnamed"):
        return str(name)
    ct = getattr(getattr(dataconfig, "metadata", None), "chain_type", "unknown")
    return str(getattr(ct, "value", ct))


def _effective_mutation_caps(dataconfigs, mutation_cap) -> dict:
    """The SHM cap ACTUALLY applied to each locus's stream, keyed by dataconfig name.

    The run-level ``mutation_cap`` is only the *request*; TCR loci are always capped to 0
    (:func:`_cap_for`). A checkpoint that recorded only the request would misstate the training
    distribution — e.g. a TRA run would read ``mutation_cap: null, heavy_shm: 0.25`` while every TRA
    stream actually saw zero SHM. Recording the effective per-locus caps makes the card reproducible."""
    out: dict = {}
    for i, dc in enumerate(dataconfigs):
        key = _dc_name(dc)
        if key in out:                                 # two panels of one locus: keep both, disambiguated
            key = f"{key}#{i}"
        out[key] = _cap_for(dc, mutation_cap)
    return out


def _mixed_stream(dataconfig, progresses, heavy_shm, seed, short_boost=1, mutation_cap=None):
    """Round-robin the amplicon-mode mix (:func:`_amplicon_specs`) so every batch spans full-length
    clean/hard + heavy-SHM + V-anchored/J-anchored/fragment short reads, all GenAIRR-native. The SHM
    cap is resolved for THIS locus (:func:`_cap_for`), so each locus in a multi-locus run gets its own
    biology."""
    specs = _amplicon_specs(progresses, heavy_shm, short_boost, _cap_for(dataconfig, mutation_cap))
    streams = [_stream_records(dataconfig, s, seed + i) for i, s in enumerate(specs)]
    yield from itertools.chain.from_iterable(zip(*streams))


def _multi_locus_stream(dataconfigs, progresses, heavy_shm, seed, short_boost=1, mutation_cap=None):
    """Interleave each locus's :func:`_mixed_stream`, tagging every record with its chain **class id**
    (the index of its locus among the DISTINCT loci, == the ``AlignAIRConfig`` chain_type order) so the
    collate builds valid chain_type/locus supervision even when two dataconfigs share a locus (e.g. two
    IGH panels) — labelling by raw dataconfig position would emit an out-of-range class.
    Reads are drawn round-robin across dataconfigs for a balanced batch."""
    from ..reference.reference_set import _locus_of
    order: list = []
    for dc in dataconfigs:
        lc = _locus_of(dc.metadata.chain_type)
        if lc not in order:
            order.append(lc)
    class_of = [order.index(_locus_of(dc.metadata.chain_type)) for dc in dataconfigs]

    def tagged(stream, ci):
        for rec in stream:
            rec["chain_type"] = ci
            yield rec
    streams = [tagged(_mixed_stream(dc, progresses, heavy_shm, seed + 1013 * i, short_boost, mutation_cap),
                      class_of[i]) for i, dc in enumerate(dataconfigs)]
    yield from itertools.chain.from_iterable(zip(*streams))


def eval_metrics(out: dict, targets: dict, cfg) -> dict:
    """Compatibility entry for the shared alignment task metrics."""
    return build_tasks(cfg).metrics(out, targets)


def train(model, reference_set, dataconfigs, cfg, logvars, *, steps=2000, batch_size=32,
          lr=3e-4, progresses=(0.3, 0.6, 0.9), heavy_shm=0.25, short_boost=1, seed=0, device="cpu",
          log_every=50, save_path=None, save_every=2000, resume_path=None,
          monitor_log=None, deep_every=500, mutation_cap=None, init_from=None, init_trust_pickle=False,
          grad_clip=None, val_every=0, val_batches=4, resume_trust_pickle=False,
          tasks: TrainingTasks | None = None):
    from .gym import Curriculum
    from .guards import validate_training_request
    tasks = build_tasks(cfg) if tasks is None else tasks
    dataconfigs = list(dataconfigs) if isinstance(dataconfigs, (list, tuple)) else [dataconfigs]
    # preflight: fail fast on a bad request rather than after a long run
    validate_training_request(steps=steps, batch_size=batch_size, lr=lr,
                              max_seq_length=cfg.max_seq_length, reference=reference_set,
                              progresses=progresses, heavy_shm=heavy_shm, short_boost=short_boost,
                              grad_clip=grad_clip)

    def _build_stream(sd):                         # single-locus mix, or interleaved multi-locus mix
        if len(dataconfigs) > 1:
            return _multi_locus_stream(dataconfigs, progresses, heavy_shm, sd, short_boost, mutation_cap)
        dc = dataconfigs[0]
        if len(progresses) > 1 or heavy_shm > 0 or short_boost > 1:
            return _mixed_stream(dc, progresses, heavy_shm, sd, short_boost, mutation_cap)
        return _stream_records(dc, _capped(dc, progresses[0]), sd)

    def _capped(dc, pr):                           # curriculum params with THIS locus's SHM cap (TCR -> 0)
        s = dict(Curriculum().params(pr))
        cap = _cap_for(dc, mutation_cap)
        if cap is not None:
            s["mutation_rate"] = min(s["mutation_rate"], cap)
        return s
    model.to(device)
    logvars.to(device)                         # Kendall log-vars must share the model's device
    opt = torch.optim.AdamW(list(model.parameters()) + list(logvars.parameters()), lr=lr)
    engine = TrainingEngine(model, opt, tasks=tasks, weights=logvars, grad_clip=grad_clip)

    start = 0
    resume_from = latest_checkpoint(resume_path) if resume_path else None
    if resume_from:
        state = load_training_checkpoint(resume_from, device=device, trust_pickle=resume_trust_pickle)
        model.load_state_dict(state.model_state)
        if state.weight_state is not None:
            logvars.load_state_dict(state.weight_state)
        if state.optimizer_state is not None:
            opt.load_state_dict(state.optimizer_state)
            for group in opt.param_groups:     # honor the requested learning rate on resume
                group["lr"] = lr
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
                   "mutation_cap": mutation_cap,
                   "effective_mutation_caps": _effective_mutation_caps(dataconfigs, mutation_cap)}
    _dcs = dataconfigs

    # ModelXRay: a read-only training X-ray over a fixed held-out probe batch
    xray = probe_input = None
    if monitor_log:
        from ..xray import ModelXRay
        ev = list(itertools.islice(_build_stream(seed + 99991), batch_size))
        probe_input, probe_targets = build_batch(ev, reference_set, cfg, device)

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
                val_set.append(build_batch(vr, reference_set, cfg, device, augment_orientation=False))
    best_score = float("-inf")

    completed = start                              # the last step actually run (a short/early
    stream = _build_stream(seed + start)           # stream must not let the final save claim `steps`)
    batch_adapter = NucleotideBatchAdapter(reference_set, cfg, device)
    for iteration in training_iterations(engine, stream, batch_adapter, start_step=start,
                                         stop_step=steps, batch_size=batch_size):
        step = completed = iteration.step
        total, parts, grad_norm = iteration.result
        if step % log_every == 0:
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
                save_checkpoint(f"{stem}.best{ext}", cfg, model, logvars, step, opt,
                                dataconfigs=_dcs, train_args={**_train_args, "val_score": vscore})
        if save_path and step % save_every == 0:
            save_rotating(save_path, cfg, model, logvars, step, opt, dataconfigs=_dcs, train_args=_train_args)
    if save_path:                                  # save the ACTUAL completed step, not the requested one
        save_rotating(save_path, cfg, model, logvars, completed, opt, dataconfigs=_dcs, train_args=_train_args)
    if xray is not None:
        xray.close()
    return model
