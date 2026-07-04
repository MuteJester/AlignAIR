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


def _rope(L, hd, base, device, offset=0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(offset, offset + L, device=device).float()      # offset for KV-cached decoding
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

    def forward(self, x, cos, sin, past_kv=None):
        B, L, _ = x.shape
        q = self.q(x).view(B, L, self.nh, self.hd).transpose(1, 2)
        k = self.k(x).view(B, L, self.nkv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, L, self.nkv, self.hd).transpose(1, 2)
        q = _apply_rope(q, cos, sin); k = _apply_rope(k, cos, sin)
        if past_kv is not None:                                      # prepend cached keys/values
            k = torch.cat([past_kv[0], k], dim=2); v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v)
        rep = self.nh // self.nkv
        kk = k.repeat_interleave(rep, dim=1); vv = v.repeat_interleave(rep, dim=1)
        # causal only on prefill (multi-token, no cache); a single cached step attends all past -> no mask
        out = F.scaled_dot_product_attention(q, kk, vv, is_causal=(past_kv is None and L > 1))
        out = out.transpose(1, 2).reshape(B, L, self.nh * self.hd)
        return self.o(out), new_kv


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

    def forward(self, x, cos, sin, past_kv=None):
        a, new_kv = self.attn(self.n1(x), cos, sin, past_kv)
        x = x + a
        x = x + self.ffn(self.n2(x))
        return x, new_kv


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

    def forward(self, input_ids, past=None, return_past=False):
        """past: per-layer (k, v) cache from previous tokens (None = full forward). With `past`,
        input_ids are only the new token(s), positioned after the cache (KV-cached incremental decode)."""
        B, L = input_ids.shape
        x = self.emb(input_ids)
        offset = 0 if past is None else past[0][0].shape[2]          # cached sequence length
        cos, sin = _rope(L, self.hd, self.cfg.rope_base, input_ids.device, offset=offset)
        cos, sin = cos[None, None].to(x.dtype), sin[None, None].to(x.dtype)
        new_past = [] if return_past else None
        for i, b in enumerate(self.blocks):
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(b, x, cos, sin, None, use_reentrant=False)[0]
            else:
                x, kv = b(x, cos, sin, None if past is None else past[i])
                if return_past:
                    new_past.append(kv)
        x = self.norm(x)
        logits = self.lm_head(x)
        return (logits, new_past) if return_past else logits

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
