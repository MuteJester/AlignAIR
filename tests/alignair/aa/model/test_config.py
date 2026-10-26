import pytest
from alignair.aa.config import aa_igh_config
from alignair.aa.tokenizer import AA_VOCAB_SIZE

pytestmark = pytest.mark.component


def test_config_is_vj_only_with_class_counts_and_aa_vocab():
    cfg = aa_igh_config(n_v_classes=149, n_j_classes=7)
    assert cfg.vocab_size == AA_VOCAB_SIZE == 23
    assert cfg.v_allele_count == 149 and cfg.j_allele_count == 7
    assert cfg.has_d is False and cfg.d_allele_count == 0
    assert cfg.max_seq_length == 192
    assert tuple(s.name for s in cfg.gene_specs) == ("v", "j")     # no D branch
    assert tuple(s.allele_count for s in cfg.gene_specs) == (149, 7)


def test_max_seq_length_leaves_room_for_the_exclusive_endpoint():
    # observed AA max is 146 residues; slot 146 must exist, so L must exceed it.
    assert aa_igh_config(149, 7).max_seq_length > 146
