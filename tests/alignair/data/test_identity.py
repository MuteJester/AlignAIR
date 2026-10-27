"""Output IDs remain unique when real headers occupy generated suffix names."""
from itertools import product

import pytest

from alignair.data.identity import UniqueSequenceIds
from alignair.errors import DataContractError

pytestmark = pytest.mark.component


@pytest.mark.parametrize("source,expected", [
    (["read", "read", "read_dup1"], ["read", "read_dup1", "read_dup1_dup1"]),
    (["read", "read_dup1", "read"], ["read", "read_dup1", "read_dup2"]),
    (["read", "read", "read", "read_dup1", "read_dup1"],
     ["read", "read_dup1", "read_dup2", "read_dup1_dup1", "read_dup1_dup2"]),
    (["seq0", "seq0", "seq0_dup1", "seq0"], ["seq0", "seq0_dup1", "seq0_dup1_dup1", "seq0_dup2"]),
])
def test_generated_names_and_real_names_share_the_same_reservation_space(source, expected):
    allocator = UniqueSequenceIds()
    assert [allocator.assign(key) for key in source] == expected


def test_id_allocation_is_unique_and_deterministic_over_adversarial_name_orders():
    for source in product(("read", "read_dup1", "read_dup2", "seq0"), repeat=4):
        first, second = UniqueSequenceIds(), UniqueSequenceIds()
        actual = [first.assign(key) for key in source]
        assert len(set(actual)) == len(source)
        assert actual == [second.assign(key) for key in source]
        assert actual[0] == source[0]


@pytest.mark.parametrize("base", [None, "", 1, False, []])
def test_invalid_names_cannot_be_reserved(base):
    allocator = UniqueSequenceIds()
    with pytest.raises(DataContractError):
        allocator.assign(base)
    assert allocator.assign("read") == "read"
