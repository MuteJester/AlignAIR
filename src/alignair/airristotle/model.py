"""Decoder-only transformer for AIRRistotle — the modern converged stack (RMSNorm pre-norm, RoPE,
SwiGLU, grouped-query attention, bias-free), causal. Architecturally a small Llama/Qwen."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        dt = x.dtype                                          # Llama computes the norm in fp32 (bf16 stability)
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.w * h.to(dt)


def _rope(L, hd, base, device):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(L, device=device).float()
    f = torch.outer(t, inv)
    emb = torch.cat([f, f], -1)
    return emb.cos(), emb.sin()


def _rotate_half(x):
    x1, x2 = x.chunk(2, -1)
    return torch.cat([-x2, x1], -1)


def _apply_rope(x, cos, sin):                                # x (B,H,L,hd)
    return x * cos + _rotate_half(x) * sin


class Attention(nn.Module):
    """Causal grouped-query attention (n_kv_heads shared across query-head groups)."""
    def __init__(self, cfg):
        super().__init__()
        self.nh, self.nkv = cfg.n_heads, cfg.n_kv_heads
        self.hd = cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * self.hd, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=False)
        self.o = nn.Linear(cfg.n_heads * self.hd, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, L, _ = x.shape
        q = self.q(x).view(B, L, self.nh, self.hd).transpose(1, 2)
        k = self.k(x).view(B, L, self.nkv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, L, self.nkv, self.hd).transpose(1, 2)
        q = _apply_rope(q, cos, sin); k = _apply_rope(k, cos, sin)
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, L, self.nh * self.hd)
        return self.o(out)


class SwiGLU(nn.Module):
    def __init__(self, d, dff):
        super().__init__()
        self.w1 = nn.Linear(d, dff, bias=False); self.w2 = nn.Linear(d, dff, bias=False)
        self.w3 = nn.Linear(dff, d, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model); self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.d_model); self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.ffn(self.n2(x))
        return x


class AIRRistotle(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.grad_checkpoint = False        # set True to trade compute for activation memory (long seqs)
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.hd = cfg.d_model // cfg.n_heads
        # Llama-style init: normal(0, init_std) on all weights; then scale the residual output
        # projections (attn.o, ffn.w3) by 1/sqrt(2*n_layers) for depth stability (GPT-2/NeoX recipe).
        self.apply(self._init_weights)
        import math
        for n, p in self.named_parameters():
            if n.endswith("attn.o.weight") or n.endswith("ffn.w3.weight"):
                p.data.mul_(1.0 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    def forward(self, input_ids):
        B, L = input_ids.shape
        x = self.emb(input_ids)
        cos, sin = _rope(L, self.hd, self.cfg.rope_base, input_ids.device)
        cos, sin = cos[None, None].to(x.dtype), sin[None, None].to(x.dtype)
        for b in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(b, x, cos, sin, use_reentrant=False)
            else:
                x = b(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def airristotle_loss(logits, batch):
    """Pure next-token cross-entropy, masked to the output span (SFT-style: no loss on the prompt).

    logits (B, L, V); batch["input_ids"] (B, L); batch["loss_mask"] (B, L) with 1 on output tokens.
    Position t's logits predict token t+1, so we score logits[:, :-1] against input_ids[:, 1:] where
    loss_mask[:, 1:] is set."""
    ids = batch["input_ids"]
    m = batch["loss_mask"][:, 1:].bool()
    if not m.any():
        return logits.new_zeros(())
    pred = logits[:, :-1][m]
    tgt = ids[:, 1:][m]
    return F.cross_entropy(pred.float(), tgt)
