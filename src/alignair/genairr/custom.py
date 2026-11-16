"""The training reference as GenAIRR dataconfigs: built-in ones by name, or one built from a user's V, D and J FASTA
files."""
from __future__ import annotations

import os

from ..errors import ConfigurationError

# The chain types whose loci carry a D gene: a custom reference of one needs a D FASTA, any other refuses one.
D_CHAIN_TYPES = frozenset({"BCR_HEAVY", "TCR_BETA", "TCR_DELTA"})


def build_dataconfigs(*, dataconfig: list | None = None, v_fasta: str | None = None,
                      d_fasta: str | None = None, j_fasta: str | None = None,
                      chain_type: str | None = None):
    """``(dataconfigs, cartridge report or None)``: the built-in dataconfigs ``dataconfig`` names, or the one GenAIRR
    builds from the V, J (and, for a locus with a D, D) FASTA files of a ``chain_type``. ConfigurationError (with the
    offending field) for both or neither, an unknown name, a missing FASTA or chain type, a D FASTA a chain type does
    not take or lacks, and a FASTA path that does not exist."""
    custom = any((v_fasta, d_fasta, j_fasta, chain_type))
    if dataconfig and custom:
        raise ConfigurationError("choose one reference: built-in dataconfigs, or custom V/J FASTA files and a chain "
                                 "type, not both", context={"field": "dataconfig"})
    if dataconfig:
        from . import resolve_dataconfigs
        return resolve_dataconfigs(dataconfig), None
    missing = [field for field, value in (("v_fasta", v_fasta), ("j_fasta", j_fasta), ("chain_type", chain_type))
               if not value]
    if missing:
        raise ConfigurationError("a custom reference needs a V FASTA, a J FASTA and a chain type (and a D FASTA for a "
                                 "locus with a D gene)", context={"field": missing[0]})
    ct = str(chain_type).upper()
    has_d = ct in D_CHAIN_TYPES
    if has_d and not d_fasta:
        raise ConfigurationError(f"chain type {ct} carries a D gene: give its D FASTA", context={"field": "d_fasta"})
    if not has_d and d_fasta:
        raise ConfigurationError(f"chain type {ct} has no D gene: give no D FASTA", context={"field": "d_fasta"})
    for field, path in (("v_fasta", v_fasta), ("j_fasta", j_fasta), ("d_fasta", d_fasta)):
        if path and not os.path.exists(path):
            raise ConfigurationError(f"the {field[0].upper()} FASTA does not exist: {path}",
                                     context={"field": field, "path": path})
    from GenAIRR.cartridge_builder import ReferenceCartridgeBuilder
    builder = ReferenceCartridgeBuilder.from_fasta(v_fasta=v_fasta, j_fasta=j_fasta, d_fasta=d_fasta,
                                                   chain_type=chain_type)
    return [builder.build()], builder.report()
