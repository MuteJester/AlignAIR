"""Shared fixtures for the SOTA detector tests: a tiny reference and a matching collated batch."""
import pytest
import torch

from alignair.reference.reference_set import ReferenceSet
from alignair.nn.sota.query_decoder import GENES


@pytest.fixture
def reference():
    return ReferenceSet.from_genotype({
        "V": {"IGHV1-1*01": "ACGTACGTACGTACGT", "IGHV1-2*01": "ACGTACGTACGTTTTT",
              "IGHV2-1*01": "TTGGCCAATTGGCCAA", "IGHV3-1*01": "GGGGCCCCAAAATTTT"},
        "D": {"IGHD1-1*01": "GGGGTTTT", "IGHD2-1*01": "CCCCAAAA", "IGHD3-1*01": "ACACACAC"},
        "J": {"IGHJ1*01": "TTTTACGTGG", "IGHJ2*01": "GGGGCACATT"},
    })


@pytest.fixture
def collated(reference):
    B, L = 3, 20
    sizes = {G: len(reference.gene(G).names) for G in GENES}
    tokens = torch.randint(1, 5, (B, L))
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[:, 16:] = False                                    # valid length 16
    out = {"tokens": tokens, "mask": mask}
    for G in GENES:
        g = G.lower()
        out[f"{g}_start"] = torch.tensor([2, 3, 4])
        out[f"{g}_end"] = torch.tensor([8, 9, 10])
        out[f"{g}_germline_start"] = torch.tensor([1, 0, 2])
        out[f"{g}_germline_end"] = torch.tensor([6, 7, 5])
        primary = torch.tensor([1, 0, 1])[:B]
        out[f"{g}_primary_idx"] = primary
        allele = torch.zeros(B, sizes[G])
        allele[torch.arange(B), primary] = 1.0
        out[f"{g}_allele"] = allele
    return out
