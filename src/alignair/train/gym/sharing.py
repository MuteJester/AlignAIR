"""Shared-state manager for PyTorch multiprocessing Gym workers."""

import numpy as np


def _pick_params(components, rng):
    """Weighted choice of one difficulty component's params (drawn once per epoch, so the
    long-run distribution is the mixture). `components` is [(weight, params), ...]."""
    weights = [max(0.0, float(c[0])) for c in components]
    tot = sum(weights)
    if tot <= 0:
        return components[0][1]
    r = float(rng.random()) * tot
    upto = 0.0
    for c, wn in zip(components, weights):
        upto += wn
        if r <= upto:
            return c[1]
    return components[-1][1]


class GymSharedState:
    """Manages shared difficulty state across multiprocessing producers using mp.Manager."""

    def __init__(self, curriculum):
        self.curriculum = curriculum
        self._version = None
        self._shared_params = None
        self._manager = None

    def enable(self) -> None:
        """Create the shared difficulty state (version flag + params dict) consumed by
        multiprocessing producers. Idempotent; only call when using num_workers>0 (it
        spawns a Manager server, so the single-process path must NOT trigger it)."""
        if self._manager is not None:
            return
        import multiprocessing as mp
        # cheap shared floor flag
        self._version = mp.Value("i", 0, lock=False)
        self._manager = mp.Manager()
        self._shared_params = self._manager.dict()
        self._shared_params["components"] = self.get_components()

    def shutdown(self) -> None:
        if self._manager is not None:
            try:
                self._manager.shutdown()
            except Exception:
                pass
            self._manager = None
            self._shared_params = None

    def __del__(self) -> None:
        self.shutdown()

    @property
    def is_enabled(self) -> bool:
        return self._shared_params is not None

    @property
    def version(self) -> int:
        return self._version.value if self._version is not None else -1

    def get_components(self, p: float = 0.0) -> list:
        if hasattr(self.curriculum, "components"):
            return self.curriculum.components()
        return [(1.0, self.curriculum.params(p))]

    def read_components(self, p: float = 0.0) -> list:
        if self.is_enabled:
            return list(self._shared_params["components"])
        return self.get_components(p)

    def push(self, p: float = 0.0) -> None:
        if not self.is_enabled:
            return
        self._shared_params["components"] = self.get_components(p)
        self._version.value += 1
