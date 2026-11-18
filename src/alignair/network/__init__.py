"""AlignAIR's network, the chip: the reference bank, the trunk, the parser, the anchors, the SHM reader, the bank scans
and D presence (``chip.ChipCore``), and its training objective (``objective.ChipObjective``). Registered with
``core.components`` as ``chip-v1``. The inference-time exact stages it calls are in ``network.refine``."""
