"""AlignAIR — a neural aligner for IG/TCR repertoires.

The stable object API (recommended):

    from alignair import Aligner
    aligner = Aligner.from_pretrained("alignair-igh-v1", device="auto")  # path / catalog id / id@rev
    result = aligner.predict(["CAGGTGCAGCTG..."])
    result.write_airr("predictions.tsv")

Train a custom model:

    from alignair import TrainingConfig, run_training
    run = run_training(TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", preset="desktop"), output_dir="runs/igh")
    aligner = run.best_aligner()

The lower-level functional façade (``load_model`` / ``predict_sequences`` / ``train_model``) remains for
back-compat. I/O + reference helpers: ``read_sequences``, ``write_airr``, ``ReferenceSet``,
``compare_airr``.
"""
from .reference.reference_set import ReferenceSet
from .io.sequence_reader import read_sequences, iter_sequences, DuplicateMetadataId
from .io.airr import write_airr, AirrWriter, PredictionCountMismatch
from .compare import compare_airr
from .api import load_model, predict_sequences, train_model
from .aligner import Aligner, PredictionResult, TrainingConfig, TrainingRun, run_training, resolve_device
from .core import AlignAIR
from .core.config import AlignAIRConfig
from .predict import PredictConfig

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("AlignAIR")
except Exception:                       # not pip-installed (e.g. run from a source tree)
    __version__ = "0+unknown"

__all__ = [
    # stable object API
    "Aligner", "PredictionResult", "TrainingConfig", "TrainingRun", "run_training", "resolve_device",
    # functional façade (back-compat)
    "load_model", "predict_sequences", "train_model",
    "AlignAIR", "AlignAIRConfig", "PredictConfig",
    "ReferenceSet", "read_sequences", "iter_sequences",
    "write_airr", "AirrWriter", "compare_airr", "DuplicateMetadataId", "PredictionCountMismatch", "__version__",
]
