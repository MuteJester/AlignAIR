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
        return self.w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


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
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.hd = cfg.d_model // cfg.n_heads
        # copy head: score each decode position against the prompt positions (attention to copy)
        self.copy_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.copy_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, input_ids):
        B, L = input_ids.shape
        x = self.emb(input_ids)
        cos, sin = _rope(L, self.hd, self.cfg.rope_base, input_ids.device)
        cos, sin = cos[None, None].to(x.dtype), sin[None, None].to(x.dtype)
        for b in self.blocks:
            x = b(x, cos, sin)
        x = self.norm(x)
        return x, self.lm_head(x)

    def copy_logits(self, hidden, prompt_len):
        q = self.copy_q(hidden)                              # (B,L,d)
        k = self.copy_k(hidden[:, :prompt_len])              # (B,P,d)
        scale = q.shape[-1] ** -0.5
        return torch.einsum("bld,bpd->blp", q, k) * scale    # (B,L,P)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def airristotle_loss(lm_logits, copy_logits, batch):
    # Causal next-token shift: hidden[t] (logits[:, t]) predicts the label at position t+1.
    lm = lm_logits[:, :-1]
    cp = copy_logits[:, :-1]
    P = cp.shape[-1]
    gen_t = batch["gen_target"][:, 1:]
    copy_t = batch["copy_target"][:, 1:].clamp(max=P - 1)
    mask = batch["loss_mask"][:, 1:].bool()
    is_copy = batch["is_copy"][:, 1:].bool() & mask
    is_gen = (~batch["is_copy"][:, 1:].bool()) & mask
    total = lm_logits.new_zeros(())
    n = mask.sum().clamp(min=1)
    if is_gen.any():
        total = total + F.cross_entropy(lm[is_gen].float(), gen_t[is_gen], reduction="sum")
    if is_copy.any():
        total = total + F.cross_entropy(cp[is_copy].float(), copy_t[is_copy], reduction="sum")
    return total / n
