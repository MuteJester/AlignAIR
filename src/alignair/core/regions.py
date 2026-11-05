"""The recombination regions a read position is labelled with (the chip's parse states read them)."""

REGIONS = ("pad", "pre", "V", "N1", "D", "N2", "J", "post")
REGION_INDEX = {name: i for i, name in enumerate(REGIONS)}
# The regions a read position can be in, without the pad label: a per-position region head's classes, in this order
# (class c is the region labelled REGION_INDEX[READ_REGIONS[c]] = c + 1).
READ_REGIONS = REGIONS[1:]
