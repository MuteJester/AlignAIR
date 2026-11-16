"""AIRR loci: the locus code of a GenAIRR chain type and of a standard allele name, a reference's locus schema
(``LocusInfo``, one block of every gene's alleles per locus) and the locus a reference without one is read as."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# GenAIRR chain-type value -> AIRR locus code
LOCUS_OF_CHAIN = {
    "BCR_HEAVY": "IGH", "BCR_LIGHT_KAPPA": "IGK", "BCR_LIGHT_LAMBDA": "IGL",
    "TCR_ALPHA": "TRA", "TCR_BETA": "TRB", "TCR_GAMMA": "TRG", "TCR_DELTA": "TRD",
}
CHAIN_OF_LOCUS = {locus: chain for chain, locus in LOCUS_OF_CHAIN.items()}
_LOCUS_NAME_RE = re.compile(r"^(IG[HKL]|TR[ABGD])")


def locus_of_chain(chain_type) -> str:
    """The AIRR locus of a GenAIRR chain type (BCR_HEAVY -> IGH); an unknown chain type is returned as given."""
    key = str(getattr(chain_type, "value", chain_type)).upper()
    return LOCUS_OF_CHAIN.get(key, key)


def locus_from_name(name: str):
    """The AIRR locus a standard IMGT/AIRR allele name belongs to (IGKV1-5*01 -> IGK), or None."""
    m = _LOCUS_NAME_RE.match(str(name).upper())
    return m.group(1) if m else None


def loci_from_membership(per_index: dict[str, list], order: list[str],
                          chain_of: dict[str, str] | None = None) -> list[LocusInfo]:
    """Build the per-locus schema from per-gene per-index locus membership. Each locus must occupy a
    single **contiguous** block within every gene head — same-locus dataconfigs merged adjacently are
    fine, but an *interleaved / split* locus (another locus's alleles inside its range) is refused
    rather than masked by a silently-widened min/max range. ``order`` is the class-id
    order; ``chain_of`` maps a locus to its GenAIRR chain-type value (defaults to the locus code)."""
    loci = []
    for lc in order:
        ranges: dict[str, tuple] = {}
        for G, loci_of in per_index.items():
            idx = [i for i, x in enumerate(loci_of) if x == lc]
            if not idx:
                continue
            s, e = idx[0], idx[-1] + 1
            if idx != list(range(s, e)):                 # not a single contiguous block -> fail closed
                raise ValueError(
                    f"locus {lc!r} alleles are not contiguous in the {G} head (interleaved / split "
                    f"loci); order the dataconfigs so each locus occupies one block, and merge "
                    f"same-locus references before building.")
            ranges[G] = (s, e)
        loci.append(LocusInfo(locus=lc, chain_type=(chain_of or {}).get(lc, lc),
                              has_d="D" in ranges, ranges=ranges))
    return loci


@dataclass
class LocusInfo:
    """One locus in a (possibly multi-chain) union: its AIRR code, GenAIRR chain type, whether it has
    a D gene, and the per-gene ``[start, end)`` index ranges it occupies in the union's allele order. A read's
    locus is the network's (``network.refine.locus``: its V, D and J calls are of it; ``predict.stages.record_loci``).
    ``shared`` names, per gene, the alleles of another
    locus's block that this locus lists too (TRAV/DV alleles, which rearrange in TRA and TRD alike): they belong to
    both loci (``ReferenceSet.locus_mask``)."""
    locus: str
    chain_type: str
    has_d: bool
    ranges: dict[str, tuple] = field(default_factory=dict)   # {"V": (s,e), "J": (s,e), ["D": (s,e)]}
    species: str | None = None                                # the species of all its alleles, when they share one
    shared: dict[str, tuple] = field(default_factory=dict)   # {"V": (name, ...)}: other loci's alleles it lists too


FALLBACK_LOCUS = "IGH"   # the locus of a reference that declares none, or several
