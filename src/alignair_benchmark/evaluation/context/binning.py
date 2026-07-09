from __future__ import annotations


def _bin(value: float, edges: tuple[float, ...], labels: tuple[str, ...]) -> str:
    for edge, label in zip(edges, labels):
        if value <= edge:
            return label
    return labels[-1]
