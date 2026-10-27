"""Compatibility imports; shared neural components live in alignair.neural."""
from ...neural.heads.region import (
    RegionTagger, decode_boundaries, REGIONS,
    REGION_INDEX,
)

__all__ = ['RegionTagger', 'decode_boundaries', 'REGIONS', 'REGION_INDEX']
