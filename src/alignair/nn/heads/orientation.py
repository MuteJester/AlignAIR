"""Compatibility imports; shared neural components live in alignair.neural."""
from ...neural.heads.orientation import (
    OrientationHead, apply_orientation, complement,
    reverse_valid, NUM_ORIENTATIONS, IDENTITY,
    REVERSE_COMPLEMENT, COMPLEMENT, REVERSE,
)

__all__ = ['OrientationHead', 'apply_orientation', 'complement', 'reverse_valid', 'NUM_ORIENTATIONS', 'IDENTITY', 'REVERSE_COMPLEMENT', 'COMPLEMENT', 'REVERSE']
