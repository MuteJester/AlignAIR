"""AlignAIR — a neural aligner for IG/TCR repertoires with a runtime (dynamic) reference.

Stable public API (use these from notebooks / pipelines):

    from alignair import load_model, predict, read_sequences, write_airr, ReferenceSet

    model = load_model("runs/my_model/bundle")
    ids, reads, _ = read_sequences("reads.fastq")
    result = predict(model, reads, genotype="donor.yaml")    # genotype optional
    result.to_airr("out.tsv", ids)

Everything under ``alignair.inference``, ``alignair.io``, ``alignair.serialization``,
``alignair.core``, ``alignair.nn``, etc. is implementation detail and may change without notice.
"""
from .api import LoadedModel, PredictionBatch, load_model, predict
from .reference.reference_set import ReferenceSet
from .io.sequence_reader import read_sequences, iter_sequences
from .io.airr import write_airr, AirrWriter
from .compare import compare_airr

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("AlignAIR")
except Exception:                       # not pip-installed (e.g. run from a source tree)
    __version__ = "0+unknown"

__all__ = [
    "load_model", "predict", "LoadedModel", "PredictionBatch",
    "ReferenceSet", "read_sequences", "iter_sequences",
    "write_airr", "AirrWriter", "compare_airr", "__version__",
]
