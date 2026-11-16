"""Prediction on the CPU spread over worker processes.

Torch's intra-op threads scale poorly on a read batch's small convolutions (the chip: 47 reads/s on one thread, 137
on eight), so N single-threaded processes, each predicting its share of a chunk, use N cores far better. Every worker
holds its own copy of the aligner and runs with one thread. Workers are forked from a forkserver: a clean process
that imports torch and alignair once (``PRELOAD``) and stays single-threaded, so a fork from it carries no thread or
lock of anyone else's; never from the parent, whose torch may have started its OpenMP pool (a fork after that can
hang the child). ``prestart`` starts the forkserver early, so its imports run while the parent loads the model; a
platform without a forkserver spawns the workers instead, each importing torch itself. The aligner is pickled once
to a private temporary file that each worker loads: handed to the workers directly it would be written down each
one's start-up pipe in turn. A chunk is cut into parts of whole batches, the batches one process runs
(``forward.bucket_batches``) dealt round-robin, a few parts per worker so a slow part does not hold the rest back;
each worker batches its part the same way, so the records are one process's, and they come back in input order. Closing
stops the workers outright: their predictions are all in by then, and each would otherwise spend half a second tearing
torch down.
"""
from __future__ import annotations

import multiprocessing
import os
import pickle
import tempfile

from .core.components import ARCHITECTURE_MODULES

PRELOAD = ("torch", "alignair.aligner")      # what the forkserver imports once for every worker
_ALIGNER = None                                   # a worker's aligner, set by _start


def _context():
    if "forkserver" not in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("spawn")
    return multiprocessing.get_context("forkserver")


def prestart() -> None:
    """Start the workers' forkserver now, importing ``PRELOAD`` and the modules ``ALIGNAIR_ARCHITECTURE_MODULES``
    names, so a ``ProcessPool`` made later forks ready workers at once. A no-op where it runs already, or where the
    platform has none."""
    context = _context()
    if context.get_start_method() != "forkserver":
        return
    extra = (part.strip() for part in os.environ.get(ARCHITECTURE_MODULES, "").split(","))
    context.set_forkserver_preload([*PRELOAD, *filter(None, extra)])
    from multiprocessing import forkserver
    forkserver.ensure_running()


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
        if workers < 1:
            raise ValueError(f"workers must be at least 1 (got {workers})")
        self.workers = workers
        self._max_seq_length = aligner.model.cfg.max_seq_length
        handle, self._path = tempfile.mkstemp(prefix="alignair-workers-", suffix=".pickle")
        try:
            with os.fdopen(handle, "wb") as out:
                pickle.dump(aligner, out, protocol=pickle.HIGHEST_PROTOCOL)
            prestart()
            self._pool = _context().Pool(workers, initializer=_start, initargs=(self._path,))
        except BaseException:
            os.unlink(self._path)
            raise

    def predict(self, sequences, *, batch_size: int, **options) -> list:
        """The records of ``sequences`` (``Aligner.predict(...).to_dicts()``, ``options`` passed on), in input
        order. The workers run the batches one process runs (``forward.bucket_batches``): each part is whole
        batches dealt round-robin, so a bucket's batches keep their order in a part, its part-full one last, and the
        worker batches them the same way; the records are the ones one process gives."""
        from .core.boundaries import read_window
        from .data.inputs import prepare_nucleotide_inputs
        from .predict.forward import bucket_batches
        sequences = list(sequences)
        if not sequences:
            return []
        reads, _ = prepare_nucleotide_inputs(sequences, read_window(self._max_seq_length))
        batches = bucket_batches(reads, max_seq_length=self._max_seq_length, batch_size=batch_size)
        parts = max(1, min(self.workers * self.PARTS_PER_WORKER, len(batches)))
        shares = [[i for batch in batches[k::parts] for i in batch] for k in range(parts)]
        jobs = [([sequences[i] for i in share], batch_size, options) for share in shares]
        records: list = [None] * len(sequences)
        for share, part in zip(shares, self._pool.starmap(_predict, jobs)):
            for i, record in zip(share, part):
                records[i] = record
        return records

    def close(self) -> None:
        """Stop the workers (every prediction has returned by now) and remove their copy of the aligner."""
        self._pool.terminate()
        self._pool.join()
        self._remove()

    def _remove(self) -> None:
        if os.path.exists(self._path):
            os.unlink(self._path)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()
