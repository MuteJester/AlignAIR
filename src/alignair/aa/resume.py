"""Compatibility imports for shared trusted research checkpoints and RNG adapters."""
from ..train.checkpoints import (save_research_checkpoint as save_checkpoint,
                                 load_research_checkpoint as load_checkpoint)
from ..train.random_state import (capture_legacy_rng as capture_rng,
                                  restore_legacy_rng as restore_rng)

__all__ = ["save_checkpoint", "load_checkpoint", "capture_rng", "restore_rng"]
