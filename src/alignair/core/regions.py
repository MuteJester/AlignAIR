"""The recombination regions a read position is labelled with (the chip's parse states read them)."""

# The regions a read position can be in, in the order the chip's parse states take them.
REGIONS = ("pre", "V", "N1", "D", "N2", "J", "post")
REGION_INDEX = {name: i for i, name in enumerate(REGIONS)}
