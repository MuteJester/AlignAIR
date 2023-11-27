from enum import Enum, auto


class NP(Enum):
    DEFAULT = auto()
    NP1_ONLY = auto()
    NP2_ONLY = auto()
    NP1_NP2 = auto()
    NONE = auto()


class TrimMode(Enum):
    DEFAULT = auto()
    NO_3_PRIME = auto()
    NO_5_PRIME = auto()
    NO_5_AND_3_PRIME = auto()

