"""Prediction on the CPU spread over worker processes.

Torch's intra-op threads scale poorly on a read batch's small convolutions (the chip: 47 reads/s on one thread, 137
on eight), so N single-threaded processes, each predicting its share of a chunk, use N cores far better. Every worker
holds its own copy of the aligner and runs with one thread; workers are spawned, never forked (a fork after torch's
OpenMP pool has run can hang the child). The aligner is pickled once to a private temporary file that each worker
loads: handed to the workers directly it would be written down each one's start-up pipe in turn, and every write
waits for that child to import torch, so N workers would start one after another. A chunk is cut into contiguous
parts, a few per worker so a slow part does not hold the rest back, and the parts' records come back in input order.
"""
from __future__ import annotations

import math
import os
import pickle
import tempfile

_ALIGNER = None                                   # a worker's aligner, set by _start


def _start(path: str) -> None:
    import torch
    global _ALIGNER
    torch.set_num_threads(1)
    with open(path, "rb") as handle:
        _ALIGNER = pickle.load(handle)


def _predict(sequences, batch_size, options) -> list:
    return _ALIGNER.predict(sequences, batch_size=batch_size, **options).to_dicts()


class ProcessPool:
    """``workers`` processes predicting with copies of ``aligner`` (a CPU :class:`~alignair.aligner.Aligner`). Use
    as a context manager, or ``close`` it."""

    PARTS_PER_WORKER = 4

    def __init__(self, aligner, workers: int):
        import multiprocessing as mp
        if workers < 1:
            raise ValueError(f"workers must be at least 1 (got {workers})")
        self.workers = workers
        handle, self._path = tempfile.mkstemp(prefix="alignair-workers-", suffix=".pickle")
        try:
            with os.fdopen(handle, "wb") as out:
                pickle.dump(aligner, out, protocol=pickle.HIGHEST_PROTOCOL)
            self._pool = mp.get_context("spawn").Pool(workers, initializer=_start, initargs=(self._path,))
        except BaseException:
            os.unlink(self._path)
            raise

    def predict(self, sequences, *, batch_size: int, **options) -> list:
        """The records of ``sequences`` (``Aligner.predict(...).to_dicts()``, ``options`` passed on), in input
        order."""
        sequences = list(sequences)
        if not sequences:
            return []
        parts = max(1, min(self.workers * self.PARTS_PER_WORKER, math.ceil(len(sequences) / batch_size)))
        size = math.ceil(len(sequences) / parts)
        jobs = [(sequences[i:i + size], batch_size, options) for i in range(0, len(sequences), size)]
        return [record for part in self._pool.starmap(_predict, jobs) for record in part]

    def close(self) -> None:
        self._pool.close()
        self._pool.join()
        self._remove()

    def _remove(self) -> None:
        if os.path.exists(self._path):
            os.unlink(self._path)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is None:
            self.close()
        else:
            self._pool.terminate()
            self._pool.join()
            self._remove()
