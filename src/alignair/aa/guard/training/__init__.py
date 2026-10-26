"""Training support for the amino-acid applicability guard.

Compute accounting and production-seed selection use only the standard library.
Tensor preparation, optimization, and study orchestration live in ``data``,
``engine``, and ``studies``; ``manifests`` binds study settings and input identities.
Callers supply data and artifact paths; importing this
package does not start training or read local research files.
"""

from .seeds import choose_production_seed

__all__ = ["choose_production_seed"]
