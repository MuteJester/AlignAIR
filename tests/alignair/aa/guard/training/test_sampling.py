"""Observe batch size, role balance, OE bounds and exact sampler-state continuation."""
from collections import Counter
import copy
import math

import pytest

from alignair.aa.guard.training.sampling import GuardBatchSampler, SAMPLING_POLICY, require_sampling_policy
from alignair.errors import ArtifactIntegrityError, ConfigurationError, DataContractError, TrainingError

pytestmark = pytest.mark.component


def pools(roles=3):
    groups = ('positive_real', 'igsf', 'structural')[:roles]
    train = [{'id': f'{group}-{i}', 'aa': 'QVQLVQSG', 'group': group} for group in groups for i in range(3)]
    oe = [{'id': f'oe-{i}', 'aa': 'VVVVVVVV'} for i in range(3)]
    return train, oe


@pytest.mark.parametrize('roles', [1, 2, 3])
@pytest.mark.parametrize('with_oe', [False, True])
@pytest.mark.parametrize('cap', [0, .15, .25, .49])
def test_batches_cover_roles_fill_slots_and_respect_cap(roles, with_oe, cap):
    train, oe = pools(roles)
    if not with_oe: oe = []
    for size in (1, 2, 3, 4, 5, 7, 10, 32, 64):
        if size - (int(size*cap) if oe else 0) < roles:
            with pytest.raises(ConfigurationError):
                GuardBatchSampler(train, oe, batch_size=size, oe_cap_fraction=cap, seed=7)
            continue
        sampler = GuardBatchSampler(train, oe, batch_size=size, oe_cap_fraction=cap, seed=7)
        totals = Counter()
        for _ in range(roles):
            main, outliers = sampler.draw()
            assert len(main) + len(outliers) == size and len(outliers)/size <= cap
            assert all(any(record is original for original in train) for record in main)
            assert all(any(record is original for original in oe) for record in outliers)
            counts = Counter(record['group'] for record in main)
            assert len(counts) == roles and max(counts.values())-min(counts.values()) <= 1
            totals.update(counts)
        assert len(set(totals.values())) == 1  # every role gets the same extras over a rotation


def test_confirmed_four_record_case_and_remainder_rotation():
    train, oe = pools(2)
    sampler = GuardBatchSampler(train, oe, batch_size=4, oe_cap_fraction=.25, seed=7)
    first, outliers = sampler.draw(); second, _ = sampler.draw()
    assert Counter(record['group'] for record in first) == {'positive_real': 2, 'igsf': 1}
    assert Counter(record['group'] for record in second) == {'positive_real': 1, 'igsf': 2}
    assert len(outliers) == 1 and sampler.state_dict()['batches_drawn'] == 2


def test_float_rounding_cannot_exceed_the_cap():
    train, oe = pools()
    cap = math.nextafter(5/12, 0.)  # multiplication rounds to 5, while 5/12 exceeds the cap
    main, outliers = GuardBatchSampler(train, oe, batch_size=12, oe_cap_fraction=cap, seed=7).draw()
    assert len(main) + len(outliers) == 12 and len(outliers) == 4 and len(outliers)/12 <= cap


def test_state_roundtrip_reproduces_records_and_rotation_counter():
    train, oe = pools()
    sampler = GuardBatchSampler(train, oe, batch_size=8, oe_cap_fraction=.15, seed=7)
    for _ in range(5): sampler.draw()
    saved = copy.deepcopy(sampler.state_dict())
    expected = [sampler.draw() for _ in range(9)]
    restored = GuardBatchSampler(train, oe, batch_size=8, oe_cap_fraction=.15, seed=7)
    restored.load_state_dict(saved, expected_batches=5)
    assert [restored.draw() for _ in range(9)] == expected
    assert restored.state_dict() == sampler.state_dict()


@pytest.mark.parametrize('changes', [{'policy': 'old'}, {'plan': {}}, {'batches_drawn': -1},
    {'batches_drawn': True}, {'batches_drawn': 1.5}, {'rng': None}, {'rng': ()}])
def test_invalid_state_rolls_back_local_rng_and_rotation(changes):
    train, oe = pools()
    sampler = GuardBatchSampler(train, oe, batch_size=8, oe_cap_fraction=.25, seed=7)
    sampler.draw(); before = sampler.state_dict()
    with pytest.raises(TrainingError): sampler.load_state_dict({**before, **changes})
    assert sampler.state_dict() == before


def test_state_progress_must_match_checkpoint_progress():
    train, oe = pools()
    sampler = GuardBatchSampler(train, oe, batch_size=8, oe_cap_fraction=.25, seed=7)
    before = sampler.state_dict()
    with pytest.raises(TrainingError): sampler.load_state_dict(before, expected_batches=3)
    assert sampler.state_dict() == before


@pytest.mark.parametrize('changes', [{'batch_size': 0}, {'batch_size': True}, {'batch_size': 1.5},
    {'batch_size': 2**10000}, {'seed': False}, {'seed': '7'}, {'oe_cap_fraction': -.1},
    {'oe_cap_fraction': .5}, {'oe_cap_fraction': float('nan')}, {'oe_cap_fraction': float('inf')},
    {'oe_cap_fraction': True}, {'oe_cap_fraction': 2**10000}])
def test_invalid_options_are_classified(changes):
    train, oe = pools()
    with pytest.raises(ConfigurationError) as caught:
        GuardBatchSampler(train, oe, **{'batch_size': 8, 'oe_cap_fraction': .25, 'seed': 7, **changes})
    assert caught.value.context['stage'] == 'guard_training_sampling'


@pytest.mark.parametrize('records', [None, {}, 'records', [None], [{'aa': ''}], [{'aa': 'A'*193}]])
def test_invalid_record_pools_are_refused(records):
    train, oe = pools()
    with pytest.raises(DataContractError): GuardBatchSampler(records, oe, batch_size=8, oe_cap_fraction=.25, seed=7)


def test_unknown_training_roles_and_empty_roles_are_refused():
    train, oe = pools()
    with pytest.raises(DataContractError):
        GuardBatchSampler([{'aa': 'QVQL', 'group': 'unknown'}], oe, batch_size=8, oe_cap_fraction=.25, seed=7)
    with pytest.raises(ConfigurationError): GuardBatchSampler([], oe, batch_size=8, oe_cap_fraction=.25, seed=7)
    with pytest.raises(DataContractError):
        GuardBatchSampler(train, [{'aa': 'A'*193}], batch_size=8, oe_cap_fraction=.25, seed=7)


def test_plan_and_state_are_independent_snapshots():
    train, oe = pools()
    sampler = GuardBatchSampler(train, oe, batch_size=8, oe_cap_fraction=.25, seed=7)
    plan = sampler.plan(); plan['role_order'].clear()
    state = sampler.state_dict(); state['plan']['role_order'].clear()
    assert len(sampler.plan()['role_order']) == 3
    assert sampler.plan()['rng_seed'] == 1007
    require_sampling_policy({'sampling_policy': SAMPLING_POLICY})
    with pytest.raises(ArtifactIntegrityError): require_sampling_policy({})
