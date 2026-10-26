"""Compatibility imports for the shared alignment objective implementations."""
from ..tasks.functional import _kendall, _mean_rows, _soft_ce, soft_gaussian_target
from ..tasks.losses import genes_of, hierarchical_loss, make_loss_weights

make_logvars = make_loss_weights

__all__ = ["genes_of", "hierarchical_loss", "make_logvars", "soft_gaussian_target"]
