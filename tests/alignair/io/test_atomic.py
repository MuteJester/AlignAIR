"""Per-file atomic publication preserves completed artifacts on writer and OS failures."""
from pathlib import Path

import pytest

from alignair.errors import InputError
from alignair.io import atomic

pytestmark = pytest.mark.component


def test_utf8_text_replaces_a_completed_file_and_creates_missing_parents(tmp_path):
    path = tmp_path / 'nested' / 'report.txt'
    atomic.atomic_write_text(path, 'first')
    atomic.atomic_write_text(path, 'complete §5\n')
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
