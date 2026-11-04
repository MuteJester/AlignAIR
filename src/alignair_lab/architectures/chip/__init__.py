"""The chip: a mechanistic aligner of separate units (bank, parser, anchors, SHM reader, bank scans, D presence),
each trained on its own GenAIRR truth (design: Notes and Plans/Architecture/2026-10-31-chip-design.md)."""
from functools import partial

from alignair.core.components import ModelComponents

from .. import register
from .core import ChipCore
from .objective import ChipObjective

register("lab-chip-v1", ModelComponents(core=ChipCore, objective=ChipObjective, reference_agnostic=True))
# the same units on a 64-channel trunk: a quarter of the trunk's multiply-adds, for CPU speed
register("lab-chipw64-v1", ModelComponents(core=partial(ChipCore, channels=64), objective=ChipObjective,
                                            reference_agnostic=True))
