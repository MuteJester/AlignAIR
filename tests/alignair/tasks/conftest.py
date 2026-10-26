"""Small hand-controlled predictions for objective and metric contracts."""
from dataclasses import replace

import pytest
import torch

from alignair.core import AlignAIRConfig
from alignair.core.boundaries import boundary_masks


@pytest.fixture
def task_batch():
    def make(protein=True, batch_size=2, auxiliary=False):
        cfg = AlignAIRConfig(v_allele_count=2, j_allele_count=2, has_d=False,
                             max_seq_length=16, use_orientation=False, use_meta_heads=False)
        if protein:
            cfg = cfg.with_amino_acid_heads()
        if auxiliary:
            cfg = replace(cfg, use_meta_heads=True, state_head=True, num_chain_types=2)
        tokens = torch.zeros(batch_size, 16, dtype=torch.long)
        tokens[:, :8] = 1
        start_mask, end_mask = boundary_masks(tokens, cfg.boundary_mode)
        out = ({"start_mask": start_mask, "end_mask": end_mask} if protein
               else {"position_mask": start_mask})
        targets = {}
        for gene in ("v", "j"):
            for boundary, value in (("start", 1.), ("end", 4.)):
                out[f"{gene}_{boundary}"] = torch.full((batch_size, 1), value, requires_grad=True)
                out[f"{gene}_{boundary}_logits"] = torch.zeros(batch_size, 16, requires_grad=True)
                targets[f"{gene}_{boundary}"] = torch.full((batch_size, 1), value)
            if protein:
                out[f"{gene}_class_logits"] = torch.zeros(batch_size, 2, requires_grad=True)
                targets[f"{gene}_class"] = torch.zeros(batch_size, dtype=torch.long)
            else:
                out[f"{gene}_allele"] = torch.full((batch_size, 2), .5, requires_grad=True)
                targets[f"{gene}_allele"] = torch.tensor([[1., 0.]]).repeat(batch_size, 1)
        if auxiliary:
            for key, value in (("mutation_rate", .2), ("indel_count", 1.), ("productive", .5)):
                out[key] = torch.full((batch_size, 1), value, requires_grad=True)
                targets[key] = torch.zeros(batch_size, 1)
            out["chain_type_logits"] = torch.zeros(batch_size, 2, requires_grad=True)
            targets["chain_type"] = torch.zeros(batch_size, dtype=torch.long)
            out["state_logits"] = torch.zeros(batch_size, 16, 4, requires_grad=True)
            targets["state_labels"] = torch.zeros(batch_size, 16, dtype=torch.long)
        return cfg, out, targets
    return make
