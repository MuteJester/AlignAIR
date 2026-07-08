"""AlignAIR — a neural aligner for IG/TCR repertoires with a runtime (dynamic) reference.

Stable I/O surface (use these from notebooks / pipelines):

    from alignair import read_sequences, write_airr, ReferenceSet, compare_airr

Model loading / prediction (``load_model``/``predict``) is being rewired onto the new ``AlignAIR``
model (``alignair.models.AlignAIR``) and its ``alignair.predict`` pipeline; the previous DNAlignAIR
product surface (api/cli/bundle) has been removed. Until the AlignAIR bundle + CLI are wired, the
stable public surface here is I/O + reference + compare.
"""
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
    "ReferenceSet", "read_sequences", "iter_sequences",
    "write_airr", "AirrWriter", "compare_airr", "__version__",
]
