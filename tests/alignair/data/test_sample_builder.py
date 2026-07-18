import numpy as np
from alignair.data.sample_builder import build_xy
from alignair.data.encoders import AlleleEncoder


def _encoder():
    enc = AlleleEncoder()
    enc.register_gene("V", ["V*01", "V*02"], sort=False)
    enc.register_gene("J", ["J*01"], sort=False)
    enc.register_gene("D", ["D*01", "Short-D"], sort=False)
    return enc


def _rec():
    return {
        "v_start": 1.0, "v_end": 10.0, "j_start": 20.0, "j_end": 30.0,
        "d_start": 12.0, "d_end": 15.0,
        "v_call_set": {"V*02"}, "j_call_set": {"J*01"}, "d_call_set": {"D*01"},
        "mutation_rate": 0.1, "indel_count": 2.0, "productive": 1.0,
    }


def test_build_xy_with_d():
    tokens = np.zeros(8, np.int64)
    x, y = build_xy(tokens, _rec(), _encoder(), has_d=True)
    assert x["tokenized_sequence"].shape == (8,)
    assert list(y["v_allele"]) == [0.0, 1.0]
    assert list(y["d_allele"]) == [1.0, 0.0]
    assert y["v_start"].tolist() == [1.0]
    assert y["indel_count"].tolist() == [2.0]
    assert set(y) >= {"v_start", "v_end", "j_start", "j_end", "d_start", "d_end",
                      "v_allele", "j_allele", "d_allele", "mutation_rate",
                      "indel_count", "productive"}


def test_build_xy_no_d_omits_d():
    enc = AlleleEncoder()
    enc.register_gene("V", ["V*01"], sort=False)
    enc.register_gene("J", ["J*01"], sort=False)
    rec = {"v_start": 0.0, "v_end": 5.0, "j_start": 6.0, "j_end": 9.0,
           "v_call_set": {"V*01"}, "j_call_set": {"J*01"},
           "mutation_rate": 0.0, "indel_count": 0.0, "productive": 0.0}
    x, y = build_xy(np.zeros(8, np.int64), rec, enc, has_d=False)
    assert "d_allele" not in y and "d_start" not in y
