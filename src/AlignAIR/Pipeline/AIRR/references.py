"""Build allele reference maps from a GenAIRR DataConfig container."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from GenAIRR.dataconfig.enums import ChainType


@dataclass(frozen=True)
class ReferenceData:
    """Immutable reference data derived from a DataConfig container."""
    v_gapped: Dict[str, str]      # V allele name → IMGT-gapped sequence (uppercase)
    d_ungapped: Dict[str, str]    # D allele name → ungapped sequence (uppercase)
    j_ungapped: Dict[str, str]    # J allele name → ungapped sequence (uppercase)
    j_anchors: Dict[str, int]     # J allele name → conserved F/W anchor position
    chain: str                     # "heavy" or "light"
    has_d: bool


def _allele_dict(allele_groups, attr: str) -> dict:
    """Flatten family-grouped allele dict into {name: attr_value}."""
    result = {}
    for group in allele_groups.values():
        for allele in group:
            val = getattr(allele, attr)
            if isinstance(val, str) and 'seq' in attr:
                val = val.upper()
            result[allele.name] = val
    return result


def build_reference_maps(dataconfig) -> ReferenceData:
    """Build reference maps from a MultiDataConfigContainer.

    Returns an immutable dataclass with all allele reference maps needed
    for AIRR alignment and formatting.
    """
    packaged = dataconfig.packaged_config()

    # Determine chain type
    has_heavy_like = (
        ChainType.BCR_HEAVY in packaged
        or (hasattr(ChainType, 'TCR_BETA') and ChainType.TCR_BETA in packaged)
    )
    has_light = (
        ChainType.BCR_LIGHT_KAPPA in packaged
        or ChainType.BCR_LIGHT_LAMBDA in packaged
    )
    chain = 'heavy' if has_heavy_like else 'light' if has_light else 'heavy'

    if chain == 'heavy':
        return _build_heavy_refs(packaged, chain)
    else:
        return _build_light_refs(packaged, chain)


def _build_heavy_refs(packaged: dict, chain: str) -> ReferenceData:
    dc = (
        packaged.get(ChainType.BCR_HEAVY)
        or (hasattr(ChainType, 'TCR_BETA') and packaged.get(ChainType.TCR_BETA))
        or next(iter(packaged.values()))
    )
    return ReferenceData(
        v_gapped=_allele_dict(dc.v_alleles, 'gapped_seq'),
        d_ungapped=_allele_dict(dc.d_alleles, 'ungapped_seq'),
        j_ungapped=_allele_dict(dc.j_alleles, 'ungapped_seq'),
        j_anchors=_allele_dict(dc.j_alleles, 'anchor'),
        chain=chain,
        has_d=True,
    )


def _build_light_refs(packaged: dict, chain: str) -> ReferenceData:
    v_map: Dict[str, str] = {}
    j_map: Dict[str, str] = {}
    j_anchors: Dict[str, int] = {}

    for ct in (ChainType.BCR_LIGHT_KAPPA, ChainType.BCR_LIGHT_LAMBDA):
        dc = packaged.get(ct)
        if dc is None:
            continue
        v_map.update(_allele_dict(dc.v_alleles, 'gapped_seq'))
        j_map.update(_allele_dict(dc.j_alleles, 'ungapped_seq'))
        j_anchors.update(_allele_dict(dc.j_alleles, 'anchor'))

    return ReferenceData(
        v_gapped=v_map,
        d_ungapped={},
        j_ungapped=j_map,
        j_anchors=j_anchors,
        chain=chain,
        has_d=False,
    )
