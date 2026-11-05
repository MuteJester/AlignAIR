"""The recombination regions a read position is labelled with (the chip's parse states read them)."""

REGIONS = ("pad", "pre", "V", "N1", "D", "N2", "J", "post")
REGION_INDEX = {name: i for i, name in enumerate(REGIONS)}
# The regions a read position can be in, without the pad label, in the order the chip's parse states take them
# (region c is the one labelled REGION_INDEX[READ_REGIONS[c]] = c + 1).
READ_REGIONS = REGIONS[1:]
