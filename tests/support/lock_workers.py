"""Spawn-safe workers for real process-contention tests."""
import time


def contend(path, backend, ready, start, active, peak, finished):
    from alignair.registry.cache import _excl_lock, _lock
    lock = _excl_lock if backend == "portable" else _lock
    ready.put(True)
    if not start.wait(15):
        raise TimeoutError("parent did not release the contention barrier")
    with lock(path):
        with active.get_lock():
            active.value += 1
            peak.value = max(peak.value, active.value)
        try:
            time.sleep(0.03)  # ensure ready contenders overlap in their acquisition attempts
        finally:
            with active.get_lock():
                active.value -= 1
    finished.put(True)
