"""Metadata uniqueness, normalization and disk-backed lifecycle contracts."""
import pytest

from alignair.errors import (
    DataContractError,
    DuplicateMetadataIdError,
    InputError,
    OperationIOError,
    current_error_context,
)
from alignair.io.metadata import MetadataIndex, build_metadata_index

pytestmark = pytest.mark.component


def _indexed(path, ids, **kwargs):
    """The rows of ``ids`` and the output columns of ``build_metadata_index(path, **kwargs)``."""
    index, columns = build_metadata_index(path, **kwargs)
    with index:
        return index.get_many(ids), columns


def test_metadata_join_carries_10x_columns(tmp_path):
    m = tmp_path / "annot.csv"
    m.write_text("contig_id,barcode,umi_count\nc1,AAAA-1,7\nc2,CCCC-1,3\n")
    meta, kept = _indexed(str(m), ["c1"], id_column="contig_id")
    assert "barcode" in kept and meta["c1"]["umi_count"] == "7"


def test_10x_metadata_normalized_to_airr_names(tmp_path):
    """A stock 10x annotations file gains AIRR-standard cell_id/umi_count/c_call (raw columns kept)."""
    m = tmp_path / "filtered_contig_annotations.csv"
    m.write_text("barcode,contig_id,reads,umis,chain,c_gene\n"
                 "AAAA-1,c1,120,7,IGH,IGHM\n")
    meta, kept = _indexed(str(m), ["c1"], id_column="contig_id", normalize_10x=True)
    assert meta["c1"]["cell_id"] == "AAAA-1"          # barcode -> cell_id
    assert meta["c1"]["umi_count"] == "7"             # umis -> umi_count
    assert meta["c1"]["c_call"] == "IGHM"             # c_gene -> c_call
    assert meta["c1"]["barcode"] == "AAAA-1"          # raw 10x column preserved
    for c in ("cell_id", "umi_count", "c_call"):
        assert c in kept


def _meta_csv(tmp_path, n):
    m = tmp_path / "meta.csv"
    lines = ["contig_id,cell_id,umi_count"]
    lines += [f"c{i},BC{i}-1,{i}" for i in range(n)]
    m.write_text("\n".join(lines) + "\n")
    return m


def test_metadata_index_get_many_batches_across_param_limit(tmp_path):
    """get_many returns the right rows and chunks past SQLite's ~999-parameter limit."""
    from alignair.io.metadata import build_metadata_index
    idx, cols = build_metadata_index(str(_meta_csv(tmp_path, 2500)), id_column="contig_id")
    try:
        ids = [f"c{i}" for i in range(2500)] + ["missing"]        # > 999 -> exercises chunking
        got = idx.get_many(ids)
        assert len(got) == 2500 and "missing" not in got
        assert got["c0"]["cell_id"] == "BC0-1" and got["c2499"]["umi_count"] == "2499"
    finally:
        idx.close()


def test_metadata_index_temp_db_deleted_on_success(tmp_path):
    from alignair.io.metadata import build_metadata_index
    import os
    idx, _ = build_metadata_index(str(_meta_csv(tmp_path, 3)), id_column="contig_id")
    path = idx._path
    assert os.path.exists(path)
    idx.close()
    assert not os.path.exists(path)                              # cleaned up after use


def test_metadata_index_duplicate_id_fails_and_cleans_up(tmp_path, monkeypatch):
    """A non-unique join key fails closed (no silent last-wins) AND leaves no orphan temp DB."""
    import tempfile
    from alignair.io.metadata import build_metadata_index, DuplicateMetadataIdError
    m = tmp_path / "dup.csv"
    m.write_text("contig_id,cell_id\nc1,A\nc1,B\n")               # c1 twice
    original_mkstemp = tempfile.mkstemp
    monkeypatch.setattr(tempfile, 'mkstemp', lambda **kwargs: original_mkstemp(dir=tmp_path, **kwargs))
    before = set(tmp_path.glob('*.alignair-meta.sqlite'))
    with pytest.raises(DuplicateMetadataIdError, match="c1"):
        build_metadata_index(str(m), id_column="contig_id")
    after = set(tmp_path.glob('*.alignair-meta.sqlite'))
    assert after == before                                       # no orphan temp DB left behind


def test_duplicate_join_keys_are_rejected_without_leaking_error_context(tmp_path):
    path = tmp_path / 'metadata.csv'
    path.write_text('id,cell_id\nread,one\nread,two\n')
    with pytest.raises(DuplicateMetadataIdError) as caught:
        build_metadata_index(path)
    assert caught.value.context['path'] == str(path)
    assert caught.value.context['record_id'] == 'read'
    assert current_error_context() == {}


def test_failed_insert_is_atomic_and_preserves_earlier_committed_batches():
    with MetadataIndex() as index:
        index.put_many([('first', {'value': 'original'})])
        for rows in ([('second', {}), ('first', {})], [('second', {}), ('second', {})]):
            with pytest.raises(DuplicateMetadataIdError) as caught:
                index.put_many(rows)
            assert caught.value.__cause__ is not None
            assert index.get_many(['first', 'second']) == {'first': {'value': 'original'}}
        index.put_many([('second', {'value': 'later'})])
        assert index.get('second') == {'value': 'later'}
        assert index.get('missing') == {} and index.get('missing', 'fallback') == 'fallback'
        assert index.get_many([]) == {}


@pytest.mark.parametrize('key', [None, '', 1, False])
def test_index_rejects_invalid_keys_before_inserting_a_partial_batch(key):
    with MetadataIndex() as index:
        with pytest.raises(DataContractError):
            index.put_many([('valid', {}), (key, {})])
        assert index.get_many(['valid']) == {}


def test_index_context_manager_closes_on_failure_and_close_is_idempotent():
    from pathlib import Path
    index = MetadataIndex()
    path = Path(index._path)
    with pytest.raises(RuntimeError, match='caller failed'):
        with index:
            raise RuntimeError('caller failed')
    assert not path.exists()
    index.close()
    for operation in (lambda: index.get_many([]), lambda: index.put_many([]), index.__enter__):
        with pytest.raises(DataContractError, match='closed'):
            operation()


@pytest.mark.parametrize('stage', ['connect', 'schema'])
def test_constructor_failure_cleans_up_connection_and_temp_file(tmp_path, monkeypatch, stage):
    import sqlite3
    import tempfile
    original_mkstemp, original_connect = tempfile.mkstemp, sqlite3.connect
    monkeypatch.setattr(tempfile, 'mkstemp', lambda **kwargs: original_mkstemp(dir=tmp_path, **kwargs))
    connections = []
    def connect(path):
        if stage == 'connect':
            raise sqlite3.OperationalError('unavailable')
        connection = original_connect(path)
        connections.append(connection)
        connection.set_authorizer(lambda *args: sqlite3.SQLITE_DENY)
        return connection
    monkeypatch.setattr(sqlite3, 'connect', connect)
    with pytest.raises(OperationIOError, match='create metadata index') as caught:
        MetadataIndex()
    assert isinstance(caught.value.__cause__, sqlite3.Error)
    assert not list(tmp_path.glob('*.alignair-meta.sqlite'))
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match='closed'):
            connection.execute('SELECT 1')


def test_interrupted_index_construction_removes_temp_file_and_propagates_cancellation(tmp_path, monkeypatch):
    import sqlite3
    import tempfile
    original_mkstemp = tempfile.mkstemp
    monkeypatch.setattr(tempfile, 'mkstemp', lambda **kwargs: original_mkstemp(dir=tmp_path, **kwargs))
    def interrupt(path): raise KeyboardInterrupt()
    monkeypatch.setattr(sqlite3, 'connect', interrupt)
    with pytest.raises(KeyboardInterrupt):
        MetadataIndex()
    assert not list(tmp_path.glob('*.alignair-meta.sqlite'))


@pytest.mark.parametrize('operation', ['insert', 'lookup'])
def test_database_failures_are_classified_and_preserve_causes(operation):
    import sqlite3
    with MetadataIndex() as index:
        forbidden = sqlite3.SQLITE_INSERT if operation == 'insert' else sqlite3.SQLITE_SELECT
        index._con.set_authorizer(lambda action, *args: sqlite3.SQLITE_DENY if action == forbidden else sqlite3.SQLITE_OK)
        with pytest.raises(OperationIOError) as caught:
            index.put_many([('read', {})]) if operation == 'insert' else index.get_many(['read'])
        assert isinstance(caught.value.__cause__, sqlite3.Error)


def test_unexpected_database_constraint_is_not_misreported_as_duplicate_id():
    with MetadataIndex() as index:
        index._con.execute("CREATE TRIGGER reject_row BEFORE INSERT ON m BEGIN SELECT RAISE(ABORT, 'blocked'); END")
        with pytest.raises(OperationIOError, match='constraint failed'):
            index.put_many([('read', {})])


@pytest.mark.parametrize('compressed', [False, True])
def test_canonical_values_win_over_10x_aliases(tmp_path, compressed):
    import gzip
    path = tmp_path / ('metadata.tsv' + ('.gz' if compressed else ''))
    text = ('\ncontig_id\tbarcode\tcell_id\tumis\tumi_count\tc_gene\n'
            'read\talias\tcanonical\t5\t9\tIGHG\n'
            'other\tfallback\t\t7\t\t\n'
            '\tignored\t\t\t\t\n')
    path.write_bytes(gzip.compress(text.encode()) if compressed else text.encode())
    expected, _ = _indexed(path, ['read', 'other'], normalize_10x=True)
    assert expected['read']['cell_id'] == 'canonical' and expected['read']['umi_count'] == '9'
    assert expected['other']['cell_id'] == 'fallback' and expected['other']['umi_count'] == '7'
    assert expected['read']['c_call'] == 'IGHG'


@pytest.mark.parametrize('text,kwargs,message', [
    ('barcode\nvalue\n', {}, 'metadata id column'),
    ('id,cell_id\nread,one\n', {'id_column': 'missing'}, 'metadata id column'),
    ('id,cell_id\nread,one\n', {'keep_columns': ['missing']}, 'metadata columns not'),
    ('id,cell_id,cell_id\nread,one,two\n', {}, 'header columns must be unique'),
    ('id,cell_id\nread\n', {}, 'missing a selected field'),
    ('id,cell_id\nread,"unterminated\n', {}, 'malformed metadata table'),
    ('"unterminated\n', {}, 'malformed metadata table'),
])
def test_bad_tables_fail_with_path_context(tmp_path, text, kwargs, message):
    path = tmp_path / 'metadata.csv'
    path.write_text(text)
    with pytest.raises(InputError, match=message) as caught:
        build_metadata_index(path, **kwargs)
    assert caught.value.context['path'] == str(path)
    assert current_error_context() == {}


@pytest.mark.parametrize('rename', [lambda col: '', lambda col: [], lambda col: 'same'])
def test_invalid_or_colliding_renames_fail_before_creating_index(tmp_path, monkeypatch, rename):
    path = tmp_path / 'metadata.csv'
    path.write_text('id,cell_id,sample_id\nread,one,two\n')
    def forbidden(): pytest.fail('created index before validating output columns')
    monkeypatch.setattr('alignair.io.metadata.MetadataIndex', forbidden)
    with pytest.raises(InputError, match='output columns'):
        build_metadata_index(path, rename=rename)


def test_rename_is_resolved_once_and_reused_for_rows(tmp_path):
    path = tmp_path / 'metadata.csv'
    path.write_text('id,cell_id\nfirst,one\nsecond,two\n')
    calls = []
    def rename(column):
        calls.append(column)
        return 'meta_' + column
    index, columns = build_metadata_index(path, rename=rename)
    with index:
        assert index.get_many(['first', 'second']) == {'first': {'meta_cell_id': 'one'}, 'second': {'meta_cell_id': 'two'}}
    assert columns == ['meta_cell_id'] and calls == ['cell_id']


def test_empty_metadata_table_with_valid_header_builds_empty_index(tmp_path):
    path = tmp_path / 'metadata.csv'
    path.write_text('id,cell_id\n')
    index, columns = build_metadata_index(path)
    with index:
        assert index.get_many(['missing']) == {} and columns == ['cell_id']


@pytest.mark.parametrize('duplicate_last', [False, True])
def test_large_metadata_input_batches_inserts_and_cleans_up_late_duplicate(tmp_path, monkeypatch, duplicate_last):
    import tempfile
    original_mkstemp = tempfile.mkstemp
    monkeypatch.setattr(tempfile, 'mkstemp', lambda **kwargs: original_mkstemp(dir=tmp_path, **kwargs))
    path = tmp_path / 'metadata.csv'
    last_id = 'read0' if duplicate_last else 'last'
    path.write_text('id,cell_id\n' + ''.join(f'read{i},cell{i}\n' for i in range(5000)) + f'{last_id},last_cell\n')
    sizes = []
    class ObservedIndex(MetadataIndex):
        def put_many(self, items):
            sizes.append(len(items))
            super().put_many(items)
    monkeypatch.setattr('alignair.io.metadata.MetadataIndex', ObservedIndex)
    if duplicate_last:
        with pytest.raises(DuplicateMetadataIdError):
            build_metadata_index(path)
    else:
        index, _ = build_metadata_index(path)
        with index:
            assert index.get('read0') == {'cell_id': 'cell0'}
            assert index.get('last') == {'cell_id': 'last_cell'}
    assert sizes == [5000, 1]
    assert not list(tmp_path.glob('*.alignair-meta.sqlite'))


def test_file_read_failures_are_typed_and_chained(tmp_path):
    import gzip
    with pytest.raises(OperationIOError) as caught:
        build_metadata_index(tmp_path / 'missing.csv')
    assert isinstance(caught.value.__cause__, FileNotFoundError)
    for name, payload in [('bad.csv', b'\xff'), ('bad.csv.gz', b'not gzip')]:
        path = tmp_path / name
        path.write_bytes(payload)
        with pytest.raises(InputError, match='cannot decode') as caught:
            build_metadata_index(path)
        assert caught.value.__cause__ is not None


def test_source_metadata_lookup_is_batched_and_ignores_absent_keys():
    from alignair.io.metadata import lookup_metadata_rows
    class Lookup:
        def get_many(self, keys):
            assert keys == ['read', 'other']
            return {'read': {'cell_id': 'one'}}
    assert lookup_metadata_rows(Lookup(), [None, 'read', '', 'read', 'other']) == [{}, {'cell_id': 'one'}, {}, {'cell_id': 'one'}, {}]


def test_source_metadata_alignment_supports_generators_and_disabled_metadata():
    from alignair.io.metadata import lookup_metadata_rows
    with MetadataIndex() as index:
        index.put_many([('read', {'cell_id': 'one'})])
        assert lookup_metadata_rows(index, iter(['read', None, '', 'missing', 'read'])) == [
            {'cell_id': 'one'}, {}, {}, {}, {'cell_id': 'one'}]
        assert lookup_metadata_rows(index, []) == []
    assert lookup_metadata_rows(None, None) is None


def test_a_metadata_index_storage_failure_is_reported_as_the_index_s_not_the_metadata_file_s(tmp_path, monkeypatch):
    """The index's own failure (a full temporary disk, an SQLite error) reaches the caller as the index's
    ``OperationIOError`` ("cannot store metadata rows"), not as "cannot read metadata input", which would send the user
    to a metadata file that is fine."""
    def full_disk(self, items):
        raise OperationIOError("cannot store metadata rows", context={"stage": "metadata_index"})
    monkeypatch.setattr(MetadataIndex, "put_many", full_disk)
    m = tmp_path / "annot.csv"
    m.write_text("contig_id,barcode\nc1,AAAA-1\n")
    with pytest.raises(OperationIOError, match="cannot store metadata rows"):
        build_metadata_index(str(m), id_column="contig_id")
