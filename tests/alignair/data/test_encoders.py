import numpy as np
from alignair.data.encoders import AlleleEncoder, ChainTypeEncoder


def test_multi_hot_single():
    enc = AlleleEncoder()
    enc.register_gene("V", ["A*01", "B*01", "C*01"], sort=False)
    ohe = enc.encode("V", [{"B*01"}])
    assert ohe.shape == (1, 3)
    assert list(ohe[0]) == [0.0, 1.0, 0.0]


def test_multi_hot_ambiguous_calls():
    enc = AlleleEncoder()
    enc.register_gene("V", ["A*01", "B*01", "C*01"], sort=False)
    ohe = enc.encode("V", [{"A*01", "C*01"}])
    assert list(ohe[0]) == [1.0, 0.0, 1.0]


def test_unknown_allele_ignored():
    enc = AlleleEncoder()
    enc.register_gene("V", ["A*01"], sort=False)
    ohe = enc.encode("V", [{"ZZZ*99"}])
    assert list(ohe[0]) == [0.0]


def test_chain_type_one_hot():
    enc = ChainTypeEncoder(["IGH", "IGK"])
    ohe = enc.encode(["IGK", "IGH"])
    assert ohe.shape == (2, 2)
    assert list(ohe[0]) == [0.0, 1.0]
    assert list(ohe[1]) == [1.0, 0.0]
