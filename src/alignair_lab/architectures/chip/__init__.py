"""The lab's chip names, kept so model files trained under them load: ``lab-chip-v1`` (a 128-channel trunk) and
``lab-chipw64-v1`` (64 channels), both the product's chip (``alignair.network``, registered there as ``chip-v1``).
``alignair convert`` turns their files into ``chip-v1`` files."""
from functools import partial

from alignair.core.components import ModelComponents
from alignair.network.chip import ChipCore
from alignair.network.objective import ChipObjective

from .. import register

register("lab-chip-v1", ModelComponents(core=ChipCore, objective=ChipObjective, reference_agnostic=True))
register("lab-chipw64-v1", ModelComponents(core=partial(ChipCore, channels=64), objective=ChipObjective,
                                            reference_agnostic=True))
