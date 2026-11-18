"""Per-file atomic publication preserves completed artifacts on writer and OS failures."""
import errno
import os
import sys
from pathlib import Path

import pytest

from alignair.errors import InputError
from alignair.io import atomic

pytestmark = pytest.mark.component


def test_a_completed_file_is_replaced_and_missing_parents_are_created(tmp_path):
    path = tmp_path / 'nested' / 'report.txt'
    atomic.atomic_write(path, lambda temporary: Path(temporary).write_text('first', encoding='utf-8'))
    atomic.atomic_write(path, lambda temporary: Path(temporary).write_text('complete §5\n', encoding='utf-8'))
    assert path.read_text(encoding='utf-8') == 'complete §5\n'
    assert list(path.parent.iterdir()) == [path]


def test_writer_sees_a_sibling_temporary_while_the_old_destination_stays_readable(tmp_path):
    path = tmp_path / 'artifact'
    path.write_bytes(b'old')
    def write(temporary):
        assert Path(temporary).parent == path.parent and Path(temporary) != path
        assert path.read_bytes() == b'old'
        Path(temporary).write_bytes(b'new')
    atomic.atomic_write(path, write)
    assert path.read_bytes() == b'new' and list(tmp_path.iterdir()) == [path]


@pytest.mark.skipif(sys.platform == 'win32', reason="Windows' own fsync refuses a read-only descriptor: the tests above")
def test_the_contents_are_flushed_through_a_descriptor_open_for_writing(tmp_path, monkeypatch):
    """Windows' fsync (``_commit``, FlushFileBuffers) needs a handle with write access and refuses a read-only one
    with EBADF, where POSIX's flushes either: an fsync that refuses a read-only descriptor, as Windows' does, still
    publishes the file."""
    import fcntl
    flush = os.fsync
    def windows_fsync(descriptor):
        if fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY:
            raise OSError(errno.EBADF, 'Bad file descriptor')
        flush(descriptor)
    monkeypatch.setattr(atomic.os, 'fsync', windows_fsync)
    path = tmp_path / 'artifact'
    atomic.atomic_write(path, lambda temporary: Path(temporary).write_bytes(b'new'))
    assert path.read_bytes() == b'new' and list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('failure', ['writer', 'fsync', 'replace'])
def test_failed_publication_keeps_the_previous_file_and_removes_temporary(tmp_path, monkeypatch, failure):
    path = tmp_path / 'artifact'
    path.write_bytes(b'completed')
    def fail(*args):
        raise OSError('injected disk failure')
    if failure != 'writer': monkeypatch.setattr(atomic.os, failure, fail)
    def write(temporary):
        Path(temporary).write_bytes(b'partial')
        if failure == 'writer': fail()
    with pytest.raises(OSError, match='disk failure'):
        atomic.atomic_write(path, write)
    assert path.read_bytes() == b'completed' and list(tmp_path.iterdir()) == [path]


def test_domain_error_inherits_requested_stage_and_final_destination(tmp_path):
    path = tmp_path / 'artifact'
    def write(_):
        raise InputError('bad payload')
    with pytest.raises(InputError) as caught:
        atomic.atomic_write(path, write, stage='study_publication')
    assert caught.value.context == {'stage': 'study_publication', 'path': str(path)}
    assert not path.exists() and not list(tmp_path.iterdir())

