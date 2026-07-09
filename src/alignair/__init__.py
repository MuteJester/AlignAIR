"""AlignAIR — a neural aligner for IG/TCR repertoires with a runtime (dynamic) reference.

Run predictions in three lines:

    from alignair import load_model, predict_sequences
    model, reference = load_model("model.pt", dataconfigs=["HUMAN_IGH_OGRDB"])
    records = predict_sequences(model, reference, ["CAGGTGCAGCTG..."])

Train a custom model:

    from alignair import train_model
    train_model(["HUMAN_IGH_OGRDB"], out_path="my_model.pt", steps=100_000)

I/O + reference helpers: ``read_sequences``, ``write_airr``, ``ReferenceSet``, ``compare_airr``.
"""
from .reference.reference_set import ReferenceSet
from .io.sequence_reader import read_sequences, iter_sequences
from .io.airr import write_airr, AirrWriter
from .compare import compare_airr
from .api import load_model, predict_sequences, train_model
from .models import AlignAIR
from .config.alignair_config import AlignAIRConfig
from .predict import PredictConfig

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("AlignAIR")
except Exception:                       # not pip-installed (e.g. run from a source tree)
    __version__ = "0+unknown"

__all__ = [
    "load_model", "predict_sequences", "train_model",
    "AlignAIR", "AlignAIRConfig", "PredictConfig",
    "ReferenceSet", "read_sequences", "iter_sequences",
    "write_airr", "AirrWriter", "compare_airr", "__version__",
]
