"""The training recipe of every screen and record cache, and of the chip baseline they are compared with: the 1.0.0
recipe's settings under the product's record recipe (``alignair.train.trainer.RECIPE``, v4 since 2026-11-09). A cache,
a baseline screen, a null band or a milestone of another recipe (every v3 one on disk) is refused, not compared: a
cache-based screen first needs a v4 cache (``cache``) and v4 baseline screens and calibration (``calibrate``).

- Data: HUMAN_IGH_OGRDB, nucleotide input, allele labels.
- Model: the chip (chip-v1) with its default options, chosen by the reference-bound model setup (``prepare_model``).
- Optimization: batch 64 at a learning rate of 3e-4, seed 0. The recipe's runs, the baseline's milestones among them,
  keep the rate constant (no schedule). A screen cools it down by default (``schedule``: cosine to ``MIN_LR``, a
  hundredth of it, over the screen's steps), so that two runs of one architecture end close to each other.
- Records (``POLICY``, the trainer's defaults): curriculum progresses 0.3, 0.6 and 0.9, heavy SHM 0.25 and short boost
  1, which make 7 components; polymerase indels in the weighted form, with 0.05 to 0.25 of reads carrying one; and
  0.75 of each full-length component's records drawn under GenAIRR's ``productive_only()``. Each component interleaves
  4 record streams: 3 selected and 1 unselected for the 4 full-length components (the three progresses and heavy SHM),
  4 unselected for each of the 3 amplicons (V-anchored, J-anchored, fragment). That is 28 streams, 12 selected and 16
  unselected, each drawing its own reserved seed block.

``training_config`` is this recipe's ``TrainingConfig``, without validation (``val_every=0``) and with the worker count
asked for: a run of the recipe may validate and use any number of data workers, and neither changes its training
batches. A schedule and an init seed may be given (``TrainingConfig.init_seed``: replicate runs on the same batches).
``recipe_train_args`` are the ``train_args`` entries that identify the recipe, and the schedule, in a checkpoint.

The retrain recipe (``retrain_config``, ``cooldown_config``) is the product run's (``chip_full_w64_5272fc63``, the
19-hour chip-v1-w64 model): 300,000 steps of batch 64 at a constant 3e-4, seed 0, validation every 2,000 steps, 7 data
workers, the flank policy's defaults and chip-v1 at its defaults, then a cooldown resumed at 300k to 320,000 steps at
3e-5; with novel alleles in its banks (``NOVEL_ALLELES``), which stop the memorization of the training alleles (the
memorization probe of 2026-11-08: a novel read's own-call 97% with them, 74% without, at 60k steps). It trains live
from GenAIRR, not from a record cache, so any seed and dataconfigs serve. Screens keep ``POLICY`` (no novel alleles),
so the lab's screens of one recipe stay comparable with each other.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

from alignair import TrainingConfig
from alignair.data.flanks import FlankPolicy
from alignair.errors import ConfigurationError
from alignair.train import CosineSchedule
from alignair.train.records.novel import NovelAlleles
from alignair.train.records.streams import CurriculumStreamPolicy
from alignair.train.trainer import RECIPE

DATACONFIGS = ("HUMAN_IGH_OGRDB",)
POLICY = CurriculumStreamPolicy(progresses=(0.3, 0.6, 0.9), heavy_shm=0.25, short_boost=1)
# The novel-allele recipe (infra Phase 2): the same mix with 40% of the eligible V genes novel per epoch of 4,096 seeds,
# SNPs 1-6; the screen ``alignair_lab.novel`` decides whether it enters the retrain.
NOVEL_ALLELES = NovelAlleles()
NOVEL_POLICY = dataclasses.replace(POLICY, novel_alleles=NOVEL_ALLELES)
SEED = 0
BATCH_SIZE = 64
LR = 3e-4
MIN_LR = LR / 100                       # where a screen's cosine schedule ends
SCHEDULES = ("cosine", "constant")
BASELINE_ARCHITECTURE = "chip-v1"
RETRAIN_STEPS = 300_000                 # the retrain (the product run): steps at the constant rate
COOLDOWN_STEPS = 320_000                # its cooldown runs on from RETRAIN_STEPS to this step, at COOLDOWN_LR
COOLDOWN_LR = 3e-5
RETRAIN_VAL_EVERY = 2000
RETRAIN_WORKERS = 7
# The checkpoints the product run kept, per phase.
RETRAIN_MILESTONES = {"full": (*range(10_000, 150_001, 10_000), 200_000, 300_000),
                      "cool": (305_000, 310_000, 315_000, 320_000)}


def schedule(name: str, steps: int) -> CosineSchedule | None:
    """The learning-rate schedule ``name`` of a run of ``steps`` steps: ``"cosine"`` decays the rate from ``LR`` to
    ``MIN_LR`` over the steps; ``"constant"``, the recipe's own, is None. ValueError for another name."""
    if name not in SCHEDULES:
        raise ValueError(f"a schedule is one of {', '.join(SCHEDULES)}, got {name!r}")
    return CosineSchedule(total_steps=steps, min_lr=MIN_LR) if name == "cosine" else None


def training_config(*, steps: int, architecture: str = BASELINE_ARCHITECTURE, model_options: Mapping | None = None,
                    device: str = "cuda", data_workers: int = 0, schedule: CosineSchedule | None = None,
                    init_seed: int | None = None, novel_alleles: NovelAlleles | None = None) -> TrainingConfig:
    """The recipe's ``TrainingConfig`` for ``steps`` steps of ``architecture`` with ``model_options``, under
    ``schedule`` (None: a constant rate) and from ``init_seed``'s initial weights (None: the seed's): with chip-v1, no
    options and no schedule, the baseline's model on the baseline's training batches. The architecture is named by
    ``architecture`` only."""
    options = dict(model_options or {})
    if "architecture" in options:
        raise ConfigurationError("name the architecture with architecture=, not as a model option",
                                 context={"field": "model_options"})
    if architecture != BASELINE_ARCHITECTURE:
        options["architecture"] = architecture
    return TrainingConfig(dataconfigs=DATACONFIGS, steps=steps, batch_size=BATCH_SIZE, lr=LR, device=device,
                          short_boost=POLICY.short_boost, seed=SEED, model_options=options,
                          data_workers=data_workers, schedule=schedule, init_seed=init_seed,
                          novel_alleles=novel_alleles)


def retrain_config(*, dataconfigs: Sequence[str] = DATACONFIGS, steps: int = RETRAIN_STEPS, seed: int = SEED,
                   schedule: CosineSchedule | None = None, val_every: int = RETRAIN_VAL_EVERY, device: str = "cuda",
                   data_workers: int = RETRAIN_WORKERS, model_options: Mapping | None = None) -> TrainingConfig:
    """The retrain recipe's ``TrainingConfig`` (see the module docstring): chip-v1 with ``model_options`` (None: its
    defaults, the product chip's), its data and initial weights from ``seed``, under ``schedule`` (None: constant)."""
    return TrainingConfig(dataconfigs=tuple(dataconfigs), steps=steps, batch_size=BATCH_SIZE, lr=LR, device=device,
                          short_boost=POLICY.short_boost, seed=seed, init_seed=None, val_every=val_every,
                          model_options=dict(model_options or {}), data_workers=data_workers, schedule=schedule,
                          flanks=FlankPolicy(), novel_alleles=NOVEL_ALLELES)


def cooldown_config(config: TrainingConfig, checkpoint: str) -> TrainingConfig:
    """``config``'s cooldown: resumed from ``checkpoint`` (its step-``RETRAIN_STEPS`` model) to ``COOLDOWN_STEPS`` at
    ``COOLDOWN_LR``."""
    return dataclasses.replace(config, steps=COOLDOWN_STEPS, lr=COOLDOWN_LR, resume_path=checkpoint)


def recipe_train_args(schedule: CosineSchedule | None = None) -> dict:
    """The ``train_args`` entries a checkpoint of this recipe records (``alignair.train.trainer._train``), as its
    model file's JSON header holds them: the record recipe, the optimization (under ``schedule``; None: the recipe's
    constant rate) and the stream policy."""
    return {"recipe": RECIPE, "batch_size": BATCH_SIZE, "lr": LR, "seed": SEED,
            "schedule": None if schedule is None else schedule.to_dict(),
            "progresses": list(POLICY.progresses), "heavy_shm": POLICY.heavy_shm, "short_boost": POLICY.short_boost,
            "mutation_cap": None, "source_mode": "auto", "indel_form": "weighted",
            "indel_fraction": list(POLICY.indel_fraction), "selected_fraction": POLICY.selected_fraction,
            "selected_shares": list(POLICY.selected_shares()), "selection_unavailable": []}
