"""The genes a read is aligned to, in read order, as the model names them (the reference names them in capitals)."""
from __future__ import annotations

GENES = ("v", "d", "j")


def model_genes(has_d: bool) -> tuple[str, ...]:
    """The genes a model calls: V, D and J, or V and J without a D."""
    return GENES if has_d else ("v", "j")
