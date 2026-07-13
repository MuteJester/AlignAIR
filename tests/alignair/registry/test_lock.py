"""P0-12: the portable atomic-lockfile protocol (`_excl_lock`) gives real cross-process mutual
exclusion where flock is absent (Windows), reclaims stale locks, and times out — replacing the old
Windows no-op that let concurrent downloads corrupt the cache. Also: env-override cache/config paths."""
import os
import threading
import time
from pathlib import Path

import pytest

from alignair.registry.cache import _excl_lock, cache_root
from alignair.registry.sources import _config_dir


def test_excl_lock_serializes_concurrent_holders(tmp_path):
    lock = tmp_path / "x.lock"
    events = []

    def worker(tag):
        with _excl_lock(lock, poll=0.005):
            events.append(("start", tag))
            time.sleep(0.05)                       # hold the lock; a second holder must wait
            events.append(("end", tag))

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    # every start is immediately followed by its own end -> no interleaving (mutual exclusion held)
    assert len(events) == 6
    for i in range(0, 6, 2):
        assert events[i][0] == "start" and events[i + 1] == ("end", events[i][1])


def test_excl_lock_reclaims_stale_lock(tmp_path):
    lock = tmp_path / "stale.lock"
    lock.write_text("99999")                       # a leftover lock from a dead holder
    os.utime(lock, (time.time() - 10_000, time.time() - 10_000))   # make it old
    with _excl_lock(lock, stale_after=1.0, timeout=2.0):
        acquired = True                            # reclaimed instead of deadlocking
    assert acquired
    assert not lock.exists()                       # released on exit


def test_excl_lock_times_out_when_held(tmp_path):
    lock = tmp_path / "held.lock"
    with _excl_lock(lock):
        with pytest.raises(TimeoutError):
            with _excl_lock(lock, timeout=0.2, poll=0.02, stale_after=1e9):
                pass                               # never reached — the outer lock is held


def test_cache_and_config_roots_honor_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("ALIGNAIR_CONFIG_DIR", str(tmp_path / "cfg"))
    assert cache_root() == Path(str(tmp_path / "c"))
    assert _config_dir() == str(tmp_path / "cfg")
