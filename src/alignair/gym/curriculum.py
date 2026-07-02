"""Compatibility facade for curriculum strategies."""

from .curriculum.base import Curriculum, _lerp
from .curriculum.stratified import StratifiedCurriculum

__all__ = ["Curriculum", "StratifiedCurriculum"]
