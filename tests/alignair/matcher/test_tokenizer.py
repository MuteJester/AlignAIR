import torch
from alignair.matcher import tokenizer as tk
from alignair.matcher.config import MatcherConfig


def test_encode_maps_bases_and_unknown_to_N():
    ids = tk.encode("ACGTNx")
    assert ids.tolist() == [1, 2, 3, 4, 5, 5]  # x -> N
    assert ids.dtype == torch.long


def test_revcomp_is_reverse_complement():
    assert tk.revcomp("AAATTTGC") == "GCAAATTT"
    assert tk.revcomp(tk.revcomp("ACGTACGT")) == "ACGTACGT"


def test_config_defaults_match_spec():
    c = MatcherConfig()
    assert (c.vocab_size, c.d_model, c.d_match, c.n_layers) == (6, 256, 128, 16)
    assert c.dilations == (1, 2, 4, 8, 16)
    assert (c.tau, c.margin, c.max_len) == (0.05, 0.2, 576)
