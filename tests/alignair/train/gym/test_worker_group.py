"""A worker group reports a worker that exited with its exit code, even when the exit code is read before the
process has been reaped."""
import os

import pytest

from alignair.errors import TrainingError
from alignair.train.gym.workers import WorkerGroup

pytestmark = pytest.mark.component


class StandIn:
    """A process seen through its sentinel. A gone one has closed its sentinel but, like a child not yet reaped,
    shows its exit code only once joined; a live one keeps its sentinel open."""

    def __init__(self, name, *, gone):
        self.name, self.pid, self.joins = name, 4242, []
        self.sentinel, writer = os.pipe()
        self._writer = None if gone else writer
        if gone:
            os.close(writer)
        self._reaped = False

    @property
    def exitcode(self):
        return 7 if self._reaped else None

    def join(self, timeout=None):
        self.joins.append(timeout)
        self._reaped = self._writer is None

    def is_alive(self):
        return False

    def release(self):
        os.close(self.sentinel)
        if self._writer is not None:
            os.close(self._writer)


def test_a_worker_gone_before_it_is_reaped_is_reported_with_its_exit_code_and_a_live_one_is_not_waited_for():
    group = WorkerGroup()
    live, gone = StandIn('alignair-records-0', gone=False), StandIn('alignair-records-1', gone=True)
    group._processes.extend([live, gone])
    try:
        with pytest.raises(TrainingError, match='exited unexpectedly') as caught:
            group.check(ended=True)
        assert caught.value.context == {'stage': 'training_record_worker', 'process': 'alignair-records-1',
                                        'exitcode': 7}
        assert live.joins == [] and len(gone.joins) == 1 and 0 < gone.joins[0] <= 5
    finally:
        group._processes.clear()
        group.close()
        live.release()
        gone.release()
