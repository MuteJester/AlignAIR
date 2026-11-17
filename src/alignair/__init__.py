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

Files: ``read_sequences`` and ``iter_sequences`` read reads (FASTA, FASTQ, a CSV/TSV table or one per line),
``write_airr`` and ``AirrWriter`` write an AIRR table. Also ``ReferenceSet`` and ``compare_airr``.

Errors: every expected failure is an ``AlignAIRError``; ``alignair.errors`` exports each of its classes.
"""
from .aligner import Aligner, PredictionResult
from .analyze.compare import compare_airr
from .errors import AlignAIRError
from .io.airr import AirrWriter, write_airr
from .io.sequence_reader import iter_sequences, read_sequences
from .reference.reference_set import ReferenceSet
from .train.config import TrainingConfig
from .train.workflow import TrainingRun, run_training

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("AlignAIR")
except Exception:  # noqa: BLE001 - not pip-installed (e.g. run from a source tree), or unreadable metadata
    __version__ = "0+unknown"

__all__ = [
    "AirrWriter",
    "AlignAIRError",
    "Aligner",
    "PredictionResult",
    "ReferenceSet",
    "TrainingConfig",
    "TrainingRun",
    "__version__",
    "compare_airr",
    "iter_sequences",
    "read_sequences",
    "run_training",
    "write_airr",
]
