import pytest
import torch
from torch.utils.data import DataLoader
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.data.experiment_presets import full_augmentation
from alignair.data.genairr import allele_vocab_from_dataconfig
from alignair.data.synthetic import SyntheticDataset
from alignair.data.collate import align_collate

L = 576


def _dataset(n=16):
    cfg = gdata.HUMAN_IGH_OGRDB
    exp = full_augmentation(cfg)
    vocab = allele_vocab_from_dataconfig(cfg)
    return SyntheticDataset(exp, max_seq_length=L, has_d=True, allele_vocab=vocab, n=n, seed=0), vocab


def test_yields_contract_samples():
    ds, vocab = _dataset(n=4)
    x, y = next(iter(ds))
    assert x["tokenized_sequence"].shape == (L,)
    assert int(x["tokenized_sequence"].max()) <= 5 and int(x["tokenized_sequence"].min()) >= 0
    assert y["v_allele"].shape == (len(vocab["V"]),)
    assert y["d_allele"].shape == (len(vocab["D"]),)
    assert set(y) >= {"v_start", "v_end", "j_start", "j_end", "d_start", "d_end",
                      "v_allele", "j_allele", "d_allele", "mutation_rate",
                      "indel_count", "productive"}


def test_dataloader_batches():
    ds, vocab = _dataset(n=16)
    dl = DataLoader(ds, batch_size=4, collate_fn=align_collate)
    x, y = next(iter(dl))
    assert x["tokenized_sequence"].shape == (4, L)
    assert x["tokenized_sequence"].dtype == torch.long
    assert y["v_allele"].shape == (4, len(vocab["V"]))


def test_finite_n_count():
    ds, _ = _dataset(n=10)
    samples = list(iter(ds))
    assert len(samples) <= 10  # may drop over-length records (rare at L=576)
    assert len(samples) >= 1
