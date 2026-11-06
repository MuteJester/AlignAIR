"""AlignAIR's one adapter to GenAIRR: resolving, listing and naming dataconfigs (``catalog``), and which GenAIRR this
process runs (``identity``). Code elsewhere in AlignAIR reaches GenAIRR's dataconfigs through these functions."""
from .catalog import dataconfig_names, list_dataconfigs, locus_of, name_of, resolve_dataconfig, resolve_dataconfigs
from .identity import genairr_identity, genairr_source_sha256

__all__ = ["dataconfig_names", "genairr_identity", "genairr_source_sha256", "list_dataconfigs", "locus_of",
           "name_of", "resolve_dataconfig", "resolve_dataconfigs"]
