import numpy as np
import pandas as pd
import torch
from alignair.config.model_config import ModelConfig
from alignair.core.single_chain import SingleChainAlignAIR
from alignair.data.dataset import allele_vocab_from_csv
from alignair.inference.predict import predict_calls

CSV = "tests/data/test/sample_igh.csv"


def test_predict_calls_on_sample_sequences(tmp_path):
    torch.manual_seed(0)
    vocab = allele_vocab_from_csv(CSV, has_d=True)
    cfg = ModelConfig(max_seq_length=576, v_allele_count=len(vocab["V"]),
                      j_allele_count=len(vocab["J"]), d_allele_count=len(vocab["D"]),
                      has_d_gene=True)
    model = SingleChainAlignAIR(cfg)
    # save + reload through the bundle to exercise the real load path
    model.save_pretrained(tmp_path)
    model = SingleChainAlignAIR.from_pretrained(tmp_path)

    sequences = pd.read_csv(CSV, nrows=5)["sequence"].tolist()
    result = predict_calls(model, sequences, allele_vocab=vocab, max_seq_length=576)

    assert len(result.v_calls) == 5
    for i, seq in enumerate(sequences):
        assert len(result.v_calls[i]) >= 1            # at least one V call
        assert len(result.d_calls[i]) >= 1 and len(result.j_calls[i]) >= 1
        assert isinstance(result.v_calls[i][0], str)
        # all coordinates within sequence bounds and each start < end
        # (cross-gene ordering is verified deterministically in test_decode.py;
        # untrained random weights can degenerate at the sequence end)
        for s, e in ((result.v_start[i], result.v_end[i]),
                     (result.d_start[i], result.d_end[i]),
                     (result.j_start[i], result.j_end[i])):
            assert 0 <= s < e <= len(seq)
        assert result.productive[i] in (True, False)


def test_predict_calls_no_d_omits_d():
    vocab = {"V": ["V*01", "V*02"], "J": ["J*01"]}
    cfg = ModelConfig(max_seq_length=256, v_allele_count=2, j_allele_count=1,
                      d_allele_count=None, has_d_gene=False)
    model = SingleChainAlignAIR(cfg)
    result = predict_calls(model, ["ACGT" * 10], allele_vocab=vocab, max_seq_length=256)
    assert result.d_calls is None
    assert len(result.v_calls) == 1
