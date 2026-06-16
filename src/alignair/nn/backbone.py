"""Sequence backbone: conv stem (local motifs) + Transformer (long-range), full
per-position resolution, right-pad + attention mask aware."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceBackbone(nn.Module):
    def __init__(self, d_model: int = 128, vocab_size: int = 6, n_layers: int = 4,
                 nhead: int = 8, dim_feedforward: int = 512,
                 stem_kernels=(7, 5), max_len: int = 1024):
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.stem = nn.ModuleList(
            [nn.Conv1d(d_model, d_model, k, padding="same") for k in stem_kernels])
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        L = tokens.shape[1]
        if L > self.max_len:
            raise ValueError(f"sequence length {L} exceeds backbone max_len {self.max_len}")
        positions = torch.arange(L, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        m = mask.unsqueeze(-1).to(x.dtype)
        x = x * m                                    # zero padded positions (clean stem input)
        h = x.transpose(1, 2)
        for conv in self.stem:
            h = F.gelu(conv(h)) * m.transpose(1, 2)
        h = h.transpose(1, 2)
        h = self.transformer(h, src_key_padding_mask=~mask)  # True = ignore (pad)
        return h * m
