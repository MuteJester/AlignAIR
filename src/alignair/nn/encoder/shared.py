"""Shared nucleotide encoder (modern attention backbone).

One encoder for BOTH reads and germline references (token-type embedding
distinguishes them), replacing the conv-stem+vanilla-Transformer backbone and the
separate shallow GermlineEncoder. Components are SOTA-standard for short biological
sequence: a small conv stem (local motifs), pre-norm Transformer blocks with rotary
position embeddings (RoPE — relative, length-generalising, better than learned abs
pos-emb for variable 50-600bp), SDPA attention (uses FlashAttention kernels on GPU),
and SwiGLU feed-forward.

forward_positions -> (B, L, d) per-position reps (masked).
forward -> (B, d) L2-normalised pooled embedding (for retrieval).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("l,f->lf", t, self.inv_freq.to(device))     # (L, hd/2)
        emb = torch.cat([freqs, freqs], dim=-1)                          # (L, hd)
        return emb.cos(), emb.sin()


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _apply_rope(x, cos, sin):  # x (B,H,L,hd); cos/sin (L,hd)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


class _Attention(nn.Module):
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.hd = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, cos, sin, key_padding_mask):
        B, L, d = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.nhead, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                  # (B,H,L,hd)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        attn_mask = key_padding_mask[:, None, None, :]                    # (B,1,1,L) True=attend
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = o.transpose(1, 2).reshape(B, L, d)
        return self.out(o)


class _SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w12 = nn.Linear(d_model, 2 * hidden)
        self.w3 = nn.Linear(hidden, d_model)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class _Block(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _Attention(d_model, nhead)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = _SwiGLU(d_model, ff_mult * d_model)

    def forward(self, x, cos, sin, key_padding_mask):
        x = x + self.attn(self.norm1(x), cos, sin, key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class SharedNucleotideEncoder(nn.Module):
    READ, GERMLINE = 0, 1

    def __init__(self, d_model: int = 256, n_layers: int = 6, nhead: int = 8,
                 vocab_size: int = 6, stem_kernels=(7, 5), max_len: int = 1024):
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.type_emb = nn.Embedding(2, d_model)                          # read / germline
        self.stem = nn.ModuleList(
            [nn.Conv1d(d_model, d_model, k, padding="same") for k in stem_kernels])
        self.rope = RotaryEmbedding(d_model // nhead)
        self.blocks = nn.ModuleList([_Block(d_model, nhead) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward_positions(self, tokens, mask, token_type: int = READ):
        L = tokens.shape[1]
        if L > self.max_len:
            raise ValueError(f"length {L} exceeds max_len {self.max_len}")
        m = mask.unsqueeze(-1).to(self.token_emb.weight.dtype)
        x = self.token_emb(tokens) + self.type_emb.weight[token_type]
        x = x * m
        h = x.transpose(1, 2)                                            # conv stem (local motifs)
        for conv in self.stem:
            h = F.gelu(conv(h)) * m.transpose(1, 2)
        x = h.transpose(1, 2)
        cos, sin = self.rope(L, tokens.device)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin, mask)
        return self.norm(x) * m

    def forward(self, tokens, mask, token_type: int = GERMLINE):
        h = self.forward_positions(tokens, mask, token_type)
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return F.normalize(self.proj(pooled), dim=-1)
