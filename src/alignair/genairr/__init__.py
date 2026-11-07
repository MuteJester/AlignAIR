"""AlignAIR's one adapter to GenAIRR: resolving, listing and naming dataconfigs (``catalog``), which GenAIRR this
process runs (``identity``), its anchor resolver (``anchors``) and a dataconfig's reference (``reference``). Code
elsewhere in AlignAIR reaches GenAIRR through these functions."""
from .anchors import ResolvedAnchor, resolve_anchor
from .catalog import dataconfig_names, list_dataconfigs, locus_of, name_of, resolve_dataconfig, resolve_dataconfigs
from .identity import genairr_identity, genairr_source_sha256
from .novel import derive_variant, novel_genotype
from .reference import reference_from_dataconfigs

__all__ = ["dataconfig_names", "derive_variant", "novel_genotype", "genairr_identity", "genairr_source_sha256", "list_dataconfigs", "locus_of",
           "name_of", "reference_from_dataconfigs", "resolve_anchor", "resolve_dataconfig", "resolve_dataconfigs",
           "ResolvedAnchor"]
