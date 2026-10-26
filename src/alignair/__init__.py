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
from .errors import AlignAIRError
from .reference.reference_set import ReferenceSet
from .io.sequence_reader import read_sequences, iter_sequences, DuplicateMetadataId
from .io.airr import write_airr, AirrWriter, PredictionCountMismatch
from .compare import compare_airr
from .api import load_model, predict_sequences, train_model
from .aligner import Aligner, PredictionResult, TrainingConfig, TrainingRun, run_training, resolve_device
from .core import AlignAIR
from .core.config import AlignAIRConfig
from .predict import PredictConfig
from .aa.product.loader import load_aa_model

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("AlignAIR")
except Exception:                       # not pip-installed (e.g. run from a source tree)
    __version__ = "0+unknown"

# module-level (model, device) -> AAPredictor cache for `predict_aa`, so repeated calls in one process
# (or one long-running session) don't re-verify + re-load the bundle every time. Deliberately separate
# from any nt-side model cache — the AA and nt inference stacks are kept fully distinct (§9).
_AA_PREDICTOR_CACHE: dict = {}


def predict_aa(sequences, *, model: str = "alignair-aa-igh-human@2.0.0", coverage_group=None,
               device: str = "auto", on_invalid: str = "error") -> list:
    """Amino-acid inference product façade (design spec §9) — DISTINCT from the nucleotide
    `predict_sequences`/`Aligner` surface above (different model family, different §6 result schema:
    no CIGAR, no productivity calls, no nucleotide coordinates).

    ``sequences`` is a single AA sequence string, an iterable of strings, or an iterable of
    ``{"sequence_id", "sequence"}`` dicts — **sequences/records only, never a file path** (§9; file
    parsing is the ``predict-aa`` CLI's job, see ``alignair.aa.product.cli``). ``model`` defaults to the
    current amino-acid model id and is resolved + verified via `load_aa_model` (cached per
    ``(model, device)`` at module scope). ``coverage_group``/``on_invalid`` are passed straight through
    to `AAPredictor.predict` (§7 / §9.1).

    Returns a plain ``list[dict]`` (already ``AAPrediction.to_dict()`` / ``ErrorResult.to_dict()``), one
    per input record in original order — JSON-serializable as-is.
    """
    key = (str(model), str(device))
    predictor = _AA_PREDICTOR_CACHE.get(key)
    if predictor is None:
        predictor = load_aa_model(model, device=device)
        _AA_PREDICTOR_CACHE[key] = predictor
    results = predictor.predict(sequences, coverage_group=coverage_group, on_invalid=on_invalid)
    return [r.to_dict() for r in results]


__all__ = [
    "AlignAIRError",
    # stable object API
    "Aligner", "PredictionResult", "TrainingConfig", "TrainingRun", "run_training", "resolve_device",
    # functional façade (back-compat)
    "load_model", "predict_sequences", "train_model",
    "AlignAIR", "AlignAIRConfig", "PredictConfig",
    "ReferenceSet", "read_sequences", "iter_sequences",
    "write_airr", "AirrWriter", "compare_airr", "DuplicateMetadataId", "PredictionCountMismatch",
    # amino-acid inference product (design spec §9) — distinct from the nucleotide surface above
    "load_aa_model", "predict_aa",
    "__version__",
]
