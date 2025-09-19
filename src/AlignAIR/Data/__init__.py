# Lightweight exports. Heavy imports are guarded to avoid optional dependency issues
try:
	from .SingleChainDataset import SingleChainDataset
	from .MultiChainDataset import MultiChainDataset
	from .MultiDataConfigContainer import MultiDataConfigContainer
except Exception:
	# Optional: these require external deps (e.g., GenAIRR). Allow submodule imports to proceed.
	SingleChainDataset = None  # type: ignore
	MultiChainDataset = None  # type: ignore
	MultiDataConfigContainer = None  # type: ignore

# Always safe to expose the reader
from .batch_readers import StreamingTableReader