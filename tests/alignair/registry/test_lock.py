"""The portable atomic-lockfile protocol (`_excl_lock`) gives real cross-process mutual
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


def test_excl_lock_treats_windows_permission_error_as_contention(tmp_path, monkeypatch):
    """On Windows, O_EXCL against a lockfile another holder has open raises PermissionError (a
    sharing violation), not FileExistsError. _excl_lock must treat that as contention and follow the
    normal poll/stale-reclaim path rather than crashing the waiting thread."""
    lock = tmp_path / "win.lock"
    lock.write_text("123")                                             # a held lockfile (exists)
    os.utime(lock, (time.time() - 10_000, time.time() - 10_000))       # ...and stale, so it reclaims

    real_open = os.open

    def fake_open(p, flags, *a, **k):
        if str(p) == str(lock) and (flags & os.O_EXCL) and os.path.exists(p):
            raise PermissionError(13, "sharing violation")             # Windows behaviour
        return real_open(p, flags, *a, **k)

    monkeypatch.setattr(os, "open", fake_open)

    acquired = False
    with _excl_lock(lock, stale_after=1.0, timeout=2.0, poll=0.01):
        acquired = True                                                # reclaimed, did not crash
    assert acquired
    assert not lock.exists()                                           # released on exit


def test_excl_lock_propagates_genuine_permission_error(tmp_path, monkeypatch):
    """A real permission failure - the lockfile cannot be created and does NOT exist - must surface,
    not be mistaken for contention and spun until timeout."""
    lock = tmp_path / "noperm.lock"                                    # does not exist
    attempts = 0

    real_open = os.open

    def fake_open(p, flags, *a, **k):
        nonlocal attempts
        if str(p) == str(lock) and (flags & os.O_EXCL):
            attempts += 1
            raise PermissionError(13, "access denied")                 # and the file is absent
        return real_open(p, flags, *a, **k)

    monkeypatch.setattr(os, "open", fake_open)

    with pytest.raises(PermissionError):                              # not TimeoutError
        with _excl_lock(lock, timeout=0.5, poll=0.01):
            pass
    assert attempts == 2                                               # one race retry, then propagate


def test_excl_lock_retries_permission_error_when_holder_just_disappeared(tmp_path, monkeypatch):
    """A Windows sharing violation can race with the holder deleting the lock before exists() runs.
    Retry that transient absence instead of misclassifying it as a genuine permission failure."""
    lock = tmp_path / "vanished.lock"
    attempts = 0
    real_open = os.open

    def fake_open(p, flags, *a, **k):
        nonlocal attempts
        if str(p) == str(lock) and (flags & os.O_EXCL):
            attempts += 1
            if attempts == 1:
                raise PermissionError(13, "sharing violation")         # holder released concurrently
        return real_open(p, flags, *a, **k)

    monkeypatch.setattr(os, "open", fake_open)

    with _excl_lock(lock, timeout=0.5, poll=0.01):
        assert lock.exists()
    assert attempts == 2
    assert not lock.exists()


def test_cache_and_config_roots_honor_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("ALIGNAIR_CONFIG_DIR", str(tmp_path / "cfg"))
    assert cache_root() == Path(str(tmp_path / "c"))
    assert _config_dir() == str(tmp_path / "cfg")
