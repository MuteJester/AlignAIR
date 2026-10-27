import pytest
import numpy as np

from alignair.data.augmentation.protein import ProteinSubstitutionPolicy, aa_indel, aa_substitute, aa_truncate, il_ambiguity
from alignair.errors import ConfigurationError

pytestmark = pytest.mark.component


def _rng():
    return np.random.default_rng(0)


def test_substitute_changes_about_the_target_fraction():
    seq = "A" * 1000
    out = aa_substitute(seq, 0.1, _rng())
    assert len(out) == len(seq)
    changed = sum(a != b for a, b in zip(seq, out))
    assert 60 <= changed <= 160                       # ~10% of 1000, with slack


def test_substitute_rate_zero_is_identity():
    seq = "QVQLVESGG"
    assert aa_substitute(seq, 0.0, _rng()) == seq


def test_indel_changes_length_by_one():
    seq = "QVQLVESGG"
    lengths = {len(aa_indel(seq, _rng())) for _ in range(20)}
    assert lengths <= {len(seq) - 1, len(seq) + 1} and lengths != {len(seq)}


def test_il_ambiguity_maps_isoleucine_and_leucine_together():
    # mass-spec cannot distinguish I and L; map both to a single symbol deterministically
    assert il_ambiguity("MILK") == il_ambiguity("MLIK")           # I<->L indistinguishable
    assert il_ambiguity("MILK")[1] == il_ambiguity("MILK")[2]     # positions 1,2 collapse
    assert il_ambiguity("QVQ") == "QVQ"                            # non-I/L untouched


def test_truncate_keeps_a_contiguous_window_within_length():
    seq = "ABCDEFGHIJ"
    out = aa_truncate(seq, min_frac=0.4, max_frac=0.8, rng=_rng())
    assert 4 <= len(out) <= 8
    assert out in [seq[i:i + len(out)] for i in range(len(seq) - len(out) + 1)]


def test_empty_corruptions_and_both_single_residue_indel_outcomes():
    assert aa_indel('',_rng())=='' and aa_truncate('',.4,.8,_rng())==''
    lengths={len(aa_indel('ACD',np.random.default_rng(seed))) for seed in range(20)}
    assert lengths=={2,4}


@pytest.mark.parametrize('fields', [{'probability': True}, {'probability': -1}, {'probability': 1.1},
    {'probability': float('nan')}, {'min_rate': '0.1'}, {'min_rate': float('inf')},
    {'max_rate': False}, {'min_rate': .3, 'max_rate': .1}])
def test_invalid_substitution_policy_is_rejected(fields):
    with pytest.raises(ConfigurationError):
        ProteinSubstitutionPolicy(**fields)


def test_substitution_policy_preserves_length_and_special_residues_and_is_seeded():
    sequence = 'ACDEFGHIKLMNPQRSTVWY' * 10 + 'X*'
    policy = ProteinSubstitutionPolicy(1, 1, 1)
    one = policy.apply(sequence, np.random.default_rng(5))
    assert one == policy.apply(sequence, np.random.default_rng(5))
    assert len(one) == len(sequence) and one != sequence and one.endswith('X*')
    assert policy.to_dict() == {'policy': 'protein-substitution-v1', 'probability': 1., 'min_rate': 1., 'max_rate': 1.}


def test_disabled_substitution_policy_does_not_consume_rng():
    first, second = np.random.default_rng(6), np.random.default_rng(6)
    assert ProteinSubstitutionPolicy(0).apply('ACD', first) == 'ACD'
    assert first.random() == second.random()
    assert ProteinSubstitutionPolicy(.1).apply('ACD', np.random.default_rng(0)) == 'ACD'
