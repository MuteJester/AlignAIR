"""The per-position edit states of a read against its germline (the chip's SHM reader labels them)."""

STATES = ("germline", "substitution", "insertion", "deletion")
STATE_INDEX = {name: i for i, name in enumerate(STATES)}
