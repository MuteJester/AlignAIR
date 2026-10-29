"""The 1.0.0 training recipe: what the conv-v1 baseline is trained with, and what every screen and record cache uses.

HUMAN_IGH_OGRDB, nucleotide input, allele labels; batch 64, a constant learning rate of 3e-4; the trainer's
curriculum (progresses 0.3, 0.6 and 0.9, heavy SHM 0.25, short boost 1); seed 0. conv-v1 runs with its defaults (no
model options); the corrected Short-D loss is automatic.
"""
from __future__ import annotations

from collections.abc import Mapping

from alignair import TrainingConfig
from alignair.errors import ConfigurationError
from alignair.train.gym.streams import CurriculumStreamPolicy

DATACONFIGS = ("HUMAN_IGH_OGRDB",)
POLICY = CurriculumStreamPolicy(progresses=(0.3, 0.6, 0.9), heavy_shm=0.25, short_boost=1)
SEED = 0
BATCH_SIZE = 64
LR = 3e-4
BASELINE_ARCHITECTURE = "conv-v1"


def training_config(*, steps: int, architecture: str = BASELINE_ARCHITECTURE, model_options: Mapping | None = None,
                    device: str = "cuda", data_workers: int = 0) -> TrainingConfig:
    """The recipe's ``TrainingConfig`` for ``steps`` steps of ``architecture`` with ``model_options``: conv-v1 with no
    options is the baseline's configuration exactly. The architecture is named by ``architecture`` only."""
    options = dict(model_options or {})
    if "architecture" in options:
        raise ConfigurationError("name the architecture with architecture=, not as a model option",
                                 context={"field": "model_options"})
    if architecture != BASELINE_ARCHITECTURE:
        options["architecture"] = architecture
    return TrainingConfig(dataconfigs=DATACONFIGS, steps=steps, batch_size=BATCH_SIZE, lr=LR, device=device,
                          short_boost=POLICY.short_boost, seed=SEED, model_options=options,
                          data_workers=data_workers)
