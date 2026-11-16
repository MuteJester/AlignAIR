"""AlignAIR — a neural aligner for IG/TCR repertoires.

The stable object API (recommended):

    from alignair import Aligner
    # a catalog id (PENDING-V4-PUBLISH), a path, or id@version
    aligner = Aligner.from_pretrained("alignair4-igh-human", device="auto")
    result = aligner.predict(["CAGGTGCAGCTG..."])
    result.write_airr("predictions.tsv")

Train a custom model:

    from alignair import TrainingConfig, run_training
    run = run_training(TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", preset="desktop"), output_dir="runs/igh")
    aligner = run.best_aligner()

The lower-level functional façade (``load_model`` / ``predict_sequences``) remains for
back-compat. I/O + reference helpers: ``read_sequences``, ``write_airr``, ``ReferenceSet``,
``compare_airr``.
"""
from .aligner import Aligner, PredictionResult, resolve_device
from .analyze.compare import compare_airr
from .api import load_model, predict_sequences
from .core import AlignAIR
from .core.config import AlignAIRConfig
from .errors import AlignAIRError
from .io.airr import AirrWriter, PredictionCountMismatchError, write_airr
from .io.metadata import DuplicateMetadataIdError
from .io.sequence_reader import iter_sequences, read_sequences
from .predict import PredictConfig
from .reference.reference_set import ReferenceSet
from .train.config import TrainingConfig
from .train.workflow import TrainingRun, run_training

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("AlignAIR")
except Exception:  # noqa: BLE001 - not pip-installed (e.g. run from a source tree), or unreadable metadata
    __version__ = "0+unknown"

__all__ = [  # noqa: RUF022 - grouped by API tier (part 2 settles the surface)
    "AlignAIRError",
    # stable object API
    "Aligner", "PredictionResult", "TrainingConfig", "TrainingRun", "run_training", "resolve_device",
    # functional façade (back-compat)
    "load_model", "predict_sequences",
    "AlignAIR", "AlignAIRConfig", "PredictConfig",
    "ReferenceSet", "read_sequences", "iter_sequences",
    "write_airr", "AirrWriter", "compare_airr", "DuplicateMetadataIdError", "PredictionCountMismatchError",
    "__version__",
]
