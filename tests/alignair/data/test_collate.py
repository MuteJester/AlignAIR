import numpy as np
import torch
from alignair.data.collate import align_collate


def test_collate_stacks_contract():
    s1 = ({"tokenized_sequence": np.array([0, 1, 2, 0], np.int64)},
          {"v_start": np.array([1.0], np.float32),
           "v_allele": np.array([1.0, 0.0], np.float32),
           "mutation_rate": np.array([0.1], np.float32)})
    s2 = ({"tokenized_sequence": np.array([0, 3, 4, 0], np.int64)},
          {"v_start": np.array([2.0], np.float32),
           "v_allele": np.array([0.0, 1.0], np.float32),
           "mutation_rate": np.array([0.2], np.float32)})
    x, y = align_collate([s1, s2])
    assert x["tokenized_sequence"].shape == (2, 4)
    assert x["tokenized_sequence"].dtype == torch.long
    assert y["v_start"].shape == (2, 1)
    assert y["v_allele"].shape == (2, 2)
    assert y["v_start"].dtype == torch.float32
