"""Parent-family assignment, stable hashes and unambiguous record identities."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from alignair.data.partitioning import partition_by_parent_lineage
from alignair.errors import ConfigurationError, DataContractError

pytestmark = pytest.mark.component


def test_parent_lineage_split_keeps_siblings_together():
    records = [{'id': 'p1', 'lineage_key': 'P1'}]
    records += [{'id': f's1_{i}', 'lineage_key': f'child-{i}', 'parent_lineage': 'P1'} for i in range(3)]
    records += [{'id': f'p{n}', 'lineage_key': f'P{n}'} for n in range(2, 20)]
    before = deepcopy(records)
    train, dev, digest = partition_by_parent_lineage(records, seed=0, dev_frac=.3)
    family = {'p1', 's1_0', 's1_1', 's1_2'}
    tr, dv = {record['id'] for record in train}, {record['id'] for record in dev}
    assert family <= tr or family <= dv
    assert tr.isdisjoint(dv) and tr | dv == {record['id'] for record in records}
    assert len(digest) == 64 and records == before
    assert all(any(row is original for original in records) for row in train + dev)


def test_known_assignments_and_hash_preserve_group_rounding_and_input_order():
    # Seed-zero SHA order for these group keys: P2, P4, P0, P3, P1.
    records = [{'id': 'r0', 'lineage_key': 'P0'}, {'id': 'r1', 'lineage_key': 'P1'},
               {'id': 'r2', 'lineage_key': 'P2'}, {'id': 'r3', 'lineage_key': 'P3'},
               {'id': 'r4', 'lineage_key': 'P4'}, {'id': 'child', 'parent_lineage': 'P2'}]
    train, dev, digest = partition_by_parent_lineage(records, seed=0, dev_frac=.5)
    assert [row['id'] for row in dev] == ['r2', 'child', 'r4']  # round(2.5) == 2 groups
    assert [row['id'] for row in train] == ['r0', 'r3', 'r1']
    expected = {'r0': 'train', 'r1': 'train', 'r2': 'dev', 'r3': 'train', 'r4': 'dev', 'child': 'dev'}
    assert digest == hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize('parent', [None, ''])
def test_empty_parent_falls_back_and_sequence_fields_are_not_required(parent):
    records = [{'id': 'first', 'lineage_key': 'family', 'parent_lineage': parent},
               {'id': 'second', 'parent_lineage': 'family', 'sequence_aa': 'QVQL'}]
    train, dev, _ = partition_by_parent_lineage(records, seed=-7)
    assert train == records and dev == []


@pytest.mark.parametrize('fraction, expected_dev', [(0, 0), (1, 2)])
def test_fraction_endpoints_and_empty_inputs_are_supported(fraction, expected_dev):
    records = [{'id': str(i), 'lineage_key': str(i)} for i in range(2)]
    train, dev, _ = partition_by_parent_lineage(iter(records), seed=np.int64(7), dev_frac=fraction)
    assert len(dev) == expected_dev and len(train) + len(dev) == 2
    assert partition_by_parent_lineage([], seed=0, dev_frac=fraction) == ([], [], hashlib.sha256(b'{}').hexdigest())


def test_reordering_input_preserves_assignments_and_digest_without_reordering_family_members():
    records = [{'id': str(i), 'lineage_key': str(i//3)} for i in range(15)]
    train, dev, digest = partition_by_parent_lineage(records, seed=42)
    other_train, other_dev, other_digest = partition_by_parent_lineage(reversed(records), seed=42)
    assert {row['id'] for row in train} == {row['id'] for row in other_train}
    assert {row['id'] for row in dev} == {row['id'] for row in other_dev} and other_digest == digest


@pytest.mark.parametrize('changes', [{'seed': True}, {'seed': 1.5}, {'seed': '42'}, {'dev_frac': True},
    {'dev_frac': -.01}, {'dev_frac': 1.01}, {'dev_frac': float('nan')}, {'dev_frac': float('inf')},
    {'dev_frac': '0.2'}, {'dev_frac': 2**10000}])
def test_invalid_options_fail_before_consuming_records(changes):
    def records():
        pytest.fail('invalid split options must not consume the input iterator')
        yield
    with pytest.raises(ConfigurationError) as caught:
        partition_by_parent_lineage(records(), **{'seed': 42, 'dev_frac': .2, **changes})
    assert caught.value.context['stage'] == 'lineage_partition'


@pytest.mark.parametrize('records', [None, 'records', b'records', {}, 1])
def test_record_collection_must_be_an_iterable_of_mappings(records):
    with pytest.raises(DataContractError): partition_by_parent_lineage(records, seed=42)


@pytest.mark.parametrize('record, field', [(None, None), ({}, 'id'), ({'id': ''}, 'id'),
    ({'id': 1, 'lineage_key': 'family'}, 'id'), ({'id': 'x'}, 'lineage_key'),
    ({'id': 'x', 'lineage_key': ''}, 'lineage_key'), ({'id': 'x', 'lineage_key': []}, 'lineage_key'),
    ({'id': 'x', 'lineage_key': 'family', 'parent_lineage': False}, 'parent_lineage')])
def test_malformed_identity_has_position_and_field_context(record, field):
    with pytest.raises(DataContractError) as caught:
        partition_by_parent_lineage([{'id': 'valid', 'lineage_key': 'first'}, record], seed=42)
    assert caught.value.context['record_index'] == 1 and caught.value.context['stage'] == 'lineage_partition'
    if field is not None: assert caught.value.context['field'] == field


def test_duplicate_ids_cannot_hide_a_record_assigned_to_another_partition():
    records = [{'id': 'same', 'lineage_key': 'P2'}, {'id': 'same', 'lineage_key': 'P0'}]
    with pytest.raises(DataContractError, match='unique') as caught:
        partition_by_parent_lineage(records, seed=0, dev_frac=.5)
    assert caught.value.context['first_record_index'] == 0 and caught.value.context['record_index'] == 1
    assert caught.value.context['record_id'] == 'same'
