import torch
from torch.utils.data import DataLoader
from alignair.data.dataset import AlignAIRDataset, allele_vocab_from_csv
from alignair.data.collate import align_collate

CSV = "tests/data/test/sample_igh.csv"


def test_vocab_from_csv_has_short_d_last():
    vocab = allele_vocab_from_csv(CSV, has_d=True)
    assert vocab["D"][-1] == "Short-D"
    assert len(vocab["V"]) == 198 and len(vocab["J"]) == 7
    # 34 unique D tokens in the CSV already include 'Short-D' (33 real + Short-D).
    assert len(vocab["D"]) == 34


def test_dataset_item_shapes():
    ds = AlignAIRDataset(CSV, max_seq_length=576, has_d=True, nrows=8)
    x, y = ds[0]
    assert x["tokenized_sequence"].shape == (576,)
    assert y["v_start"].shape == (1,)
    assert y["v_allele"].shape == (ds.v_allele_count,)
    assert y["d_allele"].shape == (ds.d_allele_count,)
    assert set(y.keys()) >= {"v_start", "v_end", "j_start", "j_end", "d_start",
                             "d_end", "v_allele", "j_allele", "d_allele",
                             "mutation_rate", "indel_count", "productive"}


def test_dataset_dataloader_batches():
    ds = AlignAIRDataset(CSV, max_seq_length=576, has_d=True, nrows=8)
    dl = DataLoader(ds, batch_size=4, collate_fn=align_collate)
    x, y = next(iter(dl))
    assert x["tokenized_sequence"].shape == (4, 576)
    assert y["v_allele"].shape == (4, ds.v_allele_count)
