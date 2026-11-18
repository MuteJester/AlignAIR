"""AlignAIR's one adapter to GenAIRR: resolving, listing and naming dataconfigs (``catalog``), which GenAIRR this
process runs (``identity``), its anchor resolver (``anchors``), a dataconfig's reference (``reference``), the
experiment reads are simulated with (``experiment``), a custom reference built from FASTA files (``custom``) and
novel alleles (``novel``). Code elsewhere in AlignAIR reaches GenAIRR through these modules, except the AIRR
assembly's amino-acid translation (``predict.airr.alignment``) and the data workers' check that they import the
parent's GenAIRR (``train.processes``)."""
from .anchors import ResolvedAnchor, resolve_anchor
from .catalog import (
    dataconfig_names,
    is_tcr,
    list_dataconfigs,
    locus_of,
    name_of,
    resolve_dataconfig,
    resolve_dataconfigs,
)
from .identity import genairr_identity, genairr_source_sha256
from .novel import derive_variant, novel_genotype
from .reference import reference_from_dataconfigs

__all__ = [
    "ResolvedAnchor",
    "dataconfig_names",
    "derive_variant",
    "genairr_identity",
    "genairr_source_sha256",
    "is_tcr",
    "list_dataconfigs",
    "locus_of",
    "name_of",
    "novel_genotype",
    "reference_from_dataconfigs",
    "resolve_anchor",
    "resolve_dataconfig",
    "resolve_dataconfigs",
]
