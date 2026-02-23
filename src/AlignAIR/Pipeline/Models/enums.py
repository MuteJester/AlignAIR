from enum import Enum


class GeneType(str, Enum):
    """The three immunoglobulin gene segments."""
    V = "v"
    D = "d"
    J = "j"


class OutputFormat(str, Enum):
    CSV = "csv"
    AIRR = "airr"


class ThresholdMethod(str, Enum):
    MAX_LIKELIHOOD_PERCENTAGE = "max_likelihood_percentage"
    CAPPED_DYNAMIC_CONFIDENCE = "capped_dynamic_confidence"
