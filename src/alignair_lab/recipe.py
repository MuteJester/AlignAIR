"""The training recipe of every screen and record cache, and of the conv-v1 baseline they are compared with: the 1.0.0
recipe's settings under the product's record recipe v3 (``alignair.train.trainer.RECIPE``).

- Data: HUMAN_IGH_OGRDB, nucleotide input, allele labels.
- Model: conv-v1 with its default options. Unknown productivity is masked and the Short-D loss is the corrected one,
  both chosen by the reference-bound model setup (``prepare_model``), never passed.
- Optimization: batch 64 at a constant learning rate of 3e-4 (no schedule), seed 0.
- Records (``POLICY``, the trainer's defaults): curriculum progresses 0.3, 0.6 and 0.9, heavy SHM 0.25 and short boost
  1, which make 7 components; polymerase indels in the weighted form, with 0.05 to 0.25 of reads carrying one; and
  0.75 of each selectable component's records drawn under GenAIRR's ``productive_only()``. Each component interleaves
  4 record streams (3 selected and 1 unselected, or 4 unselected for the V-anchored and fragment amplicons): 28
  streams, each drawing its own reserved seed block.

``training_config`` is this recipe's ``TrainingConfig``, without validation (``val_every=0``) and with the worker count
asked for: a run of the recipe may validate and use any number of data workers, and neither changes its training
batches. ``recipe_train_args`` are the ``train_args`` entries that identify the recipe in a checkpoint.
"""
from __future__ import annotations

from collections.abc import Mapping

from alignair import TrainingConfig
from alignair.errors import ConfigurationError
from alignair.train.gym.streams import CurriculumStreamPolicy
from alignair.train.trainer import RECIPE

DATACONFIGS = ("HUMAN_IGH_OGRDB",)
POLICY = CurriculumStreamPolicy(progresses=(0.3, 0.6, 0.9), heavy_shm=0.25, short_boost=1)
SEED = 0
BATCH_SIZE = 64
LR = 3e-4
BASELINE_ARCHITECTURE = "conv-v1"


def training_config(*, steps: int, architecture: str = BASELINE_ARCHITECTURE, model_options: Mapping | None = None,
                    device: str = "cuda", data_workers: int = 0) -> TrainingConfig:
    """The recipe's ``TrainingConfig`` for ``steps`` steps of ``architecture`` with ``model_options``: with conv-v1 and
    no options, the baseline's model on the baseline's training batches. The architecture is named by
    ``architecture`` only."""
    options = dict(model_options or {})
    if "architecture" in options:
        raise ConfigurationError("name the architecture with architecture=, not as a model option",
                                 context={"field": "model_options"})
    if architecture != BASELINE_ARCHITECTURE:
        options["architecture"] = architecture
    return TrainingConfig(dataconfigs=DATACONFIGS, steps=steps, batch_size=BATCH_SIZE, lr=LR, device=device,
                          short_boost=POLICY.short_boost, seed=SEED, model_options=options,
                          data_workers=data_workers)


def recipe_train_args() -> dict:
    """The ``train_args`` entries a checkpoint of this recipe records (``alignair.train.trainer.train``), as its
    model file's JSON header holds them: the record recipe, the optimization, the stream policy and the spaces."""
    return {"recipe": RECIPE, "batch_size": BATCH_SIZE, "lr": LR, "seed": SEED, "schedule": None,
            "augmentation": None, "input_space": "nucleotide", "label_space": "alleles",
            "progresses": list(POLICY.progresses), "heavy_shm": POLICY.heavy_shm, "short_boost": POLICY.short_boost,
            "mutation_cap": POLICY.mutation_cap, "source_mode": POLICY.mode, "indel_form": POLICY.indel_form,
            "indel_fraction": list(POLICY.indel_fraction), "selected_fraction": POLICY.selected_fraction,
            "selected_shares": list(POLICY.selected_shares()), "selection_unavailable": []}
