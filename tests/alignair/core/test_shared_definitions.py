"""One definition each for the token vocabulary, the read orientations and the genes a model calls."""
import pytest
import torch

from alignair.core.config import AlignAIRConfig
from alignair.reference.genes import GENES, model_genes
from alignair.core.orientations import (COMPLEMENT, FORWARD, NUM_ORIENTATIONS, ORIENTATIONS, REVERSE,
                                        REVERSE_COMPLEMENT, orient, reverse_complement)
from alignair.core.tokens import (BASES, COMPLEMENT_TOKENS, DEFAULT_WINDOW, IGNORE_INDEX, N, PAD, TOKEN_DICT,
                                  VOCAB_SIZE, encode, is_base)
from alignair.network.orientation import apply_orientation
from alignair.predict.config import PredictConfig

pytestmark = pytest.mark.component


def test_the_vocabulary_pads_with_0_reads_the_four_bases_as_1_to_4_and_anything_else_as_n():
    assert TOKEN_DICT == {"P": 0, "A": 1, "T": 2, "G": 3, "C": 4, "N": 5} and (PAD, N, VOCAB_SIZE) == (0, 5, 6)
    assert [TOKEN_DICT[base] for base in BASES] == [1, 2, 3, 4]
    assert encode("acgTNrx-").tolist() == [1, 4, 3, 2, 5, 5, 5, 5] and encode("").tolist() == []
    assert encode("A").dtype.name == "int64"
    assert is_base(torch.tensor([0, 1, 2, 3, 4, 5])).tolist() == [False, True, True, True, True, False]
    assert [COMPLEMENT_TOKENS[token] for token in encode("ATGCN")] == encode("TACGN").tolist()
    assert COMPLEMENT_TOKENS[PAD] == PAD


def test_the_label_sentinel_and_the_window_are_the_ones_every_config_uses():
    assert IGNORE_INDEX == torch.nn.CrossEntropyLoss().ignore_index
    assert AlignAIRConfig(1, 1).max_seq_length == PredictConfig().max_seq_length == DEFAULT_WINDOW == 576
    assert AlignAIRConfig(1, 1).vocab_size == VOCAB_SIZE


def test_the_four_orientations_are_named_in_the_order_of_the_orientation_head():
    assert ORIENTATIONS == ("forward", "reverse_complement", "complement", "reverse") and NUM_ORIENTATIONS == 4
    assert (FORWARD, REVERSE_COMPLEMENT, COMPLEMENT, REVERSE) == (0, 1, 2, 3)
    read = "AACGTNg"
    assert [orient(read, k) for k in range(4)] == [read, "cNACGTT", "TTGCANc", "gNTGCAA"]
    assert all(orient(orient(read, k), k) == read for k in range(4))
    assert reverse_complement("ACGGN") == "NCCGT"
    with pytest.raises(ValueError):
        orient(read, 4)


def test_the_model_orients_tokens_as_the_pipeline_orients_strings():
    read = "AACGTNGGCA"
    tokens = torch.as_tensor(encode(read)).repeat(4, 1)
    tokens = torch.cat([tokens, torch.zeros(4, 3, dtype=torch.long)], 1)          # right padding stays at the end
    oriented = apply_orientation(tokens, tokens != PAD, torch.arange(4))
    for k in range(4):
        assert oriented[k, :len(read)].tolist() == encode(orient(read, k)).tolist(), ORIENTATIONS[k]
        assert oriented[k, len(read):].tolist() == [PAD] * 3


def test_a_model_calls_v_d_and_j_or_v_and_j_without_a_d():
    assert GENES == ("v", "d", "j") and model_genes(True) == GENES and model_genes(False) == ("v", "j")
    for has_d in (True, False):
        assert AlignAIRConfig(1, 1, has_d=has_d).genes == PredictConfig(has_d=has_d).genes == model_genes(has_d)
