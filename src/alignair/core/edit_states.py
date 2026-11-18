"""The per-position edit labels of a read against its germline: the training targets (``data.targets``) and, all but
deletion (which has no read base), the classes the chip's SHM reader predicts (``network.reader.READ_EDITS``)."""

EDIT_LABELS = ("germline", "substitution", "insertion", "deletion")
EDIT_INDEX = {name: i for i, name in enumerate(EDIT_LABELS)}
