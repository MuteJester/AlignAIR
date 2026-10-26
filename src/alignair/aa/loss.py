"""Compatibility imports for categorical alignment objectives."""
from ..tasks.losses import categorical_loss as aa_loss, masked_soft_gaussian

__all__ = ["aa_loss", "masked_soft_gaussian"]
