from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from .backend import AlignResult


def align_batch(pairs, aligner, workers: int = 8) -> list[AlignResult | None]:
    """Align many (query, target) pairs, preserving input order. Threaded: the C aligners
    release the GIL, so this parallelizes across cores."""
    if not pairs:
        return []
    if workers <= 1 or len(pairs) == 1:
        return [aligner.align(q, t) for q, t in pairs]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda p: aligner.align(p[0], p[1]), pairs))
