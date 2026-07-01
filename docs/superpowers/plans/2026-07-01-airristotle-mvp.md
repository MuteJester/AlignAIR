# AIRRistotle MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the training scaffolding for AIRRistotle and prove the core hypothesis — that a small decoder-only LLM can learn IG V(D)J annotation by **copying** the right allele + coordinates out of a genotype-in-prompt — by overfitting a tiny GenAIRR set to near-zero loss.

**Architecture:** New isolated package `src/alignair/airristotle/`: a char-level DNA tokenizer, a prompt-builder that turns a GenAIRR record into `(input_ids, targets)` where calls/coords are **copy-pointer targets into the prompt**, a decoder-only transformer (RMSNorm/RoPE/SwiGLU/GQA — reusing the RoPE/SwiGLU patterns already in `nn/encoder/shared.py`) with an LM head + a **copy head** (attention over prompt positions), and a two-channel (generate | copy) loss + training loop.

**Tech Stack:** PyTorch, GenAIRR, bf16, AdamW. Venv: `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python`. GPU: RTX 3090Ti 24GB.

## Global Constraints

- Run via `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python` and `... -m pytest`. Never bare `python`.
- **Isolation:** everything under `src/alignair/airristotle/` + `scripts/train_airristotle.py`. Do NOT modify the current XAttnAligner/DNAlignAIR models, inference, or training. Reuse the gym + reference set read-only.
- **Decoder-only** (Llama/Qwen-style): RMSNorm pre-norm, RoPE, SwiGLU, grouped-query attention, bias-free, bf16. Same modern stack as frontier LLMs.
- **Copy mechanism for calls + coordinates** — never generate coordinate digits; a copy head selects a prompt position (exact, single-source).
- **Starting size ~150M params**; scale only if learning bottlenecks.
- Char-level DNA tokenization (small vocab, long sequences).
- Git commit messages: never add Co-Authored-By/Claude attribution. Do not touch `.claude/settings.json` or `docs/architecture/adoption_roadmap.md`.
- MVP uses a SMALL genotype in the prompt (true alleles + a few distractors) — NOT the full 198-allele reference (keeps context ~few-k tokens). Dynamic-genotype/novel-allele sampling is a LATER plan.

## File Structure

- `src/alignair/airristotle/__init__.py` — package exports.
- `src/alignair/airristotle/tokenizer.py` — `AIRRTokenizer` (char-level DNA + special/structural tokens).
- `src/alignair/airristotle/prompt.py` — `Example` dataclass + `build_example(...)` (record → ids + copy/gen targets + loss mask).
- `src/alignair/airristotle/config.py` — `AIRRConfig` dataclass (~150M dims).
- `src/alignair/airristotle/model.py` — `AIRRistotle` decoder-only transformer + copy head; `airristotle_loss(...)`.
- `scripts/train_airristotle.py` — data batching + training loop + overfit sanity.
- Tests under `tests/alignair/airristotle/`.

---

### Task 1: Char-level DNA tokenizer

**Files:**
- Create: `src/alignair/airristotle/__init__.py`, `src/alignair/airristotle/tokenizer.py`
- Test: `tests/alignair/airristotle/test_tokenizer.py`

**Interfaces:**
- Produces: `AIRRTokenizer()` with `.vocab: dict[str,int]`, `.vocab_size: int`, `.id(tok:str)->int`, `.encode(tokens:list[str])->list[int]`, `.decode(ids:list[int])->list[str]`, and the special-token constants as attributes (`PAD, GENO, READ, ANNOT, ORI, PROD, V, D, J, S, EOS` and field markers `VS,VE,VGS,VGE,DS,DE,DGS,DGE,JS,JE,JGS,JGE,JNS,JNE`). Tokens are STRINGS; a "token" is either a single DNA char, a special marker, or an allele-name string (allele names are added to the vocab lazily is NOT allowed — instead allele names are emitted as per-character tokens too; see prompt.py). Digits `0..9`, `+`(fwd), `-`(rc/complement flags) included.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/airristotle/test_tokenizer.py
from alignair.airristotle.tokenizer import AIRRTokenizer


def test_dna_and_specials_roundtrip():
    tok = AIRRTokenizer()
    seq = [tok.GENO, tok.V, "A", "C", "G", "T", "N", tok.S, tok.READ, "G", "A", tok.EOS]
    ids = tok.encode(seq)
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == seq


def test_vocab_has_dna_specials_digits():
    tok = AIRRTokenizer()
    for c in "ACGTN0123456789+-":
        assert c in tok.vocab
    for s in (tok.PAD, tok.GENO, tok.READ, tok.ANNOT, tok.V, tok.D, tok.J, tok.S,
              tok.ORI, tok.PROD, tok.EOS, tok.VS, tok.JNE):
        assert s in tok.vocab
    assert tok.id(tok.PAD) == 0                      # PAD must be 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/airristotle/test_tokenizer.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# src/alignair/airristotle/__init__.py
from .tokenizer import AIRRTokenizer

__all__ = ["AIRRTokenizer"]
```

```python
# src/alignair/airristotle/tokenizer.py
"""Char-level DNA tokenizer for AIRRistotle. A 'token' is a single DNA base, a digit, a flag, or a
named special/field marker. Small vocab (~50), long sequences — the deliberate tradeoff for exact
copy-pointer coordinates over a genotype-in-prompt."""
from __future__ import annotations

_DNA = list("ACGTN")
_DIGITS = list("0123456789")
_FLAGS = list("+-")
_SPECIALS = ["<PAD>", "<GENO>", "<READ>", "<ANNOT>", "<S>", "<EOS>", "<ORI>", "<PROD>",
             "<V>", "<D>", "<J>",
             "<VS>", "<VE>", "<VGS>", "<VGE>",
             "<DS>", "<DE>", "<DGS>", "<DGE>",
             "<JS>", "<JE>", "<JGS>", "<JGE>",
             "<JNS>", "<JNE>"]


class AIRRTokenizer:
    def __init__(self):
        toks = ["<PAD>"] + [s for s in _SPECIALS if s != "<PAD>"] + _DNA + _DIGITS + _FLAGS
        self.vocab = {t: i for i, t in enumerate(toks)}
        self.inv = {i: t for t, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        # expose specials as attributes: PAD, GENO, ... and field markers VS, VE, ...
        for s in _SPECIALS:
            setattr(self, s.strip("<>"), s)

    def id(self, tok: str) -> int:
        return self.vocab[tok]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.vocab[t] for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.inv[int(i)] for i in ids]
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/airristotle/test_tokenizer.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/airristotle/__init__.py src/alignair/airristotle/tokenizer.py tests/alignair/airristotle/test_tokenizer.py
git commit -m "airristotle: char-level DNA tokenizer (specials + fields + digits)"
```

---

### Task 2: Prompt-builder with copy-pointer targets (the crux)

**Files:**
- Create: `src/alignair/airristotle/prompt.py`
- Test: `tests/alignair/airristotle/test_prompt.py`

**Interfaces:**
- Consumes: `AIRRTokenizer` (Task 1); a GenAIRR `record` dict (keys incl. `sequence`, `{v,d,j}_call`, `{v,d,j}_sequence_start/end`, `{v,d,j}_germline_start/end`, `junction_start/end`, `productive`); a `ReferenceSet` (`.gene("V").names/.sequences`).
- Produces:
  `@dataclass Example: input_ids:list[int]; gen_target:list[int]; copy_target:list[int]; is_copy:list[int]; loss_mask:list[int]; prompt_len:int` (all sequence-aligned to input_ids; targets are next-token, i.e. position t predicts token t+1). And
  `build_example(record, reference_set, tokenizer, n_distractors:int=8, rng=random.Random) -> Example`.
- Semantics: `input_ids` = PROMPT (genotype allele blocks + read) then ANNOTATION. For each annotation position where `loss_mask==1`: if `is_copy==1`, the label is `copy_target` (an index INTO input_ids, must be < prompt_len — the pointed-to prompt token); else the label is `gen_target` (a vocab id). Non-loss positions have mask 0.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/airristotle/test_prompt.py
import random, pytest
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.airristotle.tokenizer import AIRRTokenizer
from alignair.airristotle.prompt import build_example
from alignair.reference.reference_set import ReferenceSet
from alignair.gym.gym import build_experiment


def _clean_record():
    exp = build_experiment(gdata.HUMAN_IGH_OGRDB, dict(
        mutation_rate=0.0, productive_only=False, end_loss_5=(0, 0), end_loss_3=(0, 0),
        indel_count=(0, 0), seq_error_rate=0.0, ambiguous_count=(0, 0)))
    return list(exp.stream_records(n=1, seed=1))[0]


def test_copy_target_for_v_start_points_at_true_read_position():
    tok = AIRRTokenizer(); rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    rec = _clean_record()
    ex = build_example(rec, rs, tok, n_distractors=4, rng=random.Random(0))
    # find the annotation step whose gen_target is the <VS> marker; the NEXT loss step is the copy for v_start
    ids = ex.input_ids
    vs_marker = tok.id(tok.VS)
    # the position emitting <VS> is a generate step; the following copy step's copy_target is a read position
    steps = [t for t in range(len(ids)) if ex.loss_mask[t]]
    # locate a copy step that should equal true v_sequence_start (read index within the READ block)
    # read block start index in the prompt:
    read_tok = tok.id(tok.READ)
    read_block_start = ids.index(read_tok) + 1                 # first base after <READ>
    true_vs = int(rec["v_sequence_start"])
    target_prompt_pos = read_block_start + true_vs
    assert any(ex.is_copy[t] and ex.copy_target[t] == target_prompt_pos for t in steps)


def test_copy_target_for_v_call_points_at_true_allele_block():
    tok = AIRRTokenizer(); rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    rec = _clean_record()
    ex = build_example(rec, rs, tok, n_distractors=4, rng=random.Random(0))
    # the v_call copy target must be < prompt_len and land on a <V> marker position
    v_marker = tok.id(tok.V)
    copy_calls = [ex.copy_target[t] for t in range(len(ex.input_ids))
                  if ex.loss_mask[t] and ex.is_copy[t] and ex.input_ids[ex.copy_target[t]] == v_marker]
    assert copy_calls                                          # at least one copy points to a <V> block


def test_prompt_len_and_masks_consistent():
    tok = AIRRTokenizer(); rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    ex = build_example(_clean_record(), rs, tok, n_distractors=4, rng=random.Random(0))
    n = len(ex.input_ids)
    assert len(ex.gen_target) == len(ex.copy_target) == len(ex.is_copy) == len(ex.loss_mask) == n
    assert all(ex.copy_target[t] < ex.prompt_len for t in range(n) if ex.loss_mask[t] and ex.is_copy[t])
    assert sum(ex.loss_mask) > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/airristotle/test_prompt.py -q`
Expected: FAIL (`build_example` missing).

- [ ] **Step 3: Implement**

```python
# src/alignair/airristotle/prompt.py
"""Turn a GenAIRR record into an AIRRistotle training example: a genotype-in-prompt + read, followed
by the annotation, where CALLS and COORDINATES are copy-pointer targets INTO the prompt (exact,
single-source). MVP: small genotype = the true V/D/J alleles + a few distractors (no novel-allele /
dynamic sampling yet)."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Example:
    input_ids: list          # prompt tokens then annotation tokens (ids)
    gen_target: list         # next-token vocab id (used when is_copy==0 and loss_mask==1)
    copy_target: list        # index into input_ids (< prompt_len) (used when is_copy==1)
    is_copy: list            # 1 if this step's label is a copy-pointer, else 0
    loss_mask: list          # 1 if this step contributes to the loss (annotation region), else 0
    prompt_len: int


def _allele_block(tokens, positions, tok, marker, name, seq):
    """Append '<marker> <name-as-chars> <S> <seq-as-chars>' to tokens; record the marker position."""
    marker_pos = len(tokens)
    tokens.append(marker)
    tokens.extend(list(name))                     # allele name char tokens (all chars are in vocab? see note)
    tokens.append(tok.S)
    seq_start = len(tokens)
    tokens.extend(list(seq.upper()))
    return marker_pos, seq_start


def build_example(record, reference_set, tokenizer, n_distractors: int = 8, rng=None):
    import random as _random
    rng = rng or _random.Random(0)
    tok = tokenizer
    seq = str(record["sequence"]).upper()
    genes = {"v": ("V", tok.V), "d": ("D", tok.D), "j": ("J", tok.J)}
    # ---- build the PROMPT: genotype allele blocks + read ----
    tokens = [tok.GENO]
    call_marker_pos = {}                          # gene -> prompt index of the TRUE allele's <marker>
    germ_block_start = {}                         # gene -> prompt index where that allele's seq starts
    for g, (G, marker) in genes.items():
        true_call = str(record.get(f"{g}_call") or "").split(",")[0]
        gene = reference_set.gene(G)
        if true_call not in gene.names:
            continue
        names = [true_call]
        pool = [n for n in gene.names if n != true_call]
        rng.shuffle(pool)
        names += pool[:n_distractors]
        rng.shuffle(names)                        # true allele not always first
        for nm in names:
            s = gene.sequences[gene.names.index(nm)]
            mpos, sstart = _allele_block(tokens, None, tok, marker, nm, s)
            if nm == true_call:
                call_marker_pos[g] = mpos
                germ_block_start[g] = sstart
    tokens.append(tok.READ)
    read_block_start = len(tokens)
    tokens.extend(list(seq))
    prompt_len = len(tokens)

    # ---- build the ANNOTATION (targets) ----
    # helper to append a generate step (emit a vocab token) or a copy step (point into prompt)
    gen_t, copy_t, is_copy, loss = [], [], [], []
    def emit_gen(t):                              # generate vocab token t next
        tokens.append(t); gen_t.append(t); copy_t.append(0); is_copy.append(0); loss.append(1)
    def emit_copy(prompt_pos, placeholder):       # copy step: label = prompt_pos; input token = placeholder
        tokens.append(placeholder); gen_t.append(0); copy_t.append(int(prompt_pos)); is_copy.append(1); loss.append(1)

    emit_gen(tok.ANNOT)
    emit_gen(tok.ORI); emit_gen("+")              # MVP: forward reads only
    for g, (G, marker) in genes.items():
        if g not in call_marker_pos:
            continue
        emit_gen(marker)
        emit_copy(call_marker_pos[g], marker)     # CALL = copy the true allele's <marker> position
        # coordinates: in-sequence start/end (read positions), germline start/end (germline positions)
        fields = [(f"{G}S", read_block_start + int(record[f"{g}_sequence_start"])),
                  (f"{G}E", read_block_start + int(record[f"{g}_sequence_end"])),
                  (f"{G}GS", germ_block_start[g] + int(record[f"{g}_germline_start"])),
                  (f"{G}GE", germ_block_start[g] + int(record[f"{g}_germline_end"]))]
        for fmark, ppos in fields:
            if 0 <= ppos < prompt_len:
                emit_gen(getattr(tok, fmark)); emit_copy(ppos, getattr(tok, fmark))
    # junction (read positions)
    if record.get("junction_start") is not None:
        emit_gen(tok.JNS); emit_copy(read_block_start + int(record["junction_start"]), tok.JNS)
        emit_gen(tok.JNE); emit_copy(read_block_start + int(record["junction_end"]), tok.JNE)
    emit_gen(tok.PROD); emit_gen("1" if record.get("productive") else "0")
    emit_gen(tok.EOS)

    # prompt positions carry NO loss; the arrays above are annotation-only, so left-pad with prompt zeros
    n_prompt = prompt_len
    gen_target = [0] * n_prompt + gen_t
    copy_target = [0] * n_prompt + copy_t
    is_copy_full = [0] * n_prompt + is_copy
    loss_full = [0] * n_prompt + loss
    # NOTE: targets are aligned so that position (prompt_len-1 .. end-1) predicts the annotation tokens.
    # shift handled in the loss (position t predicts label[t]); see training.
    return Example(input_ids=tokens, gen_target=gen_target, copy_target=copy_target,
                   is_copy=is_copy_full, loss_mask=loss_full, prompt_len=prompt_len)
```

**Note on allele-name chars:** allele names contain letters/digits/`*`/`-` (e.g. `IGHVF1-G1*01`). The MVP tokenizer only has `ACGTN0-9+-`. Before implementing, extend `_SPECIALS`/vocab in `tokenizer.py` to also include the uppercase letters `A-Z` and `*` so names tokenize (add `list("BEFHIKLMOPQRUVWXYZ")` — the non-DNA uppercase letters — and `"*"` to the vocab in `AIRRTokenizer.__init__`). Update `test_tokenizer.py` is not required, but re-run it to confirm no regression.

- [ ] **Step 4: Extend tokenizer vocab for allele-name chars, then run**

Edit `tokenizer.py` `__init__`: add `_NAMECHARS = list("BEFHIKLMOPQRUVWXYZ*")` and include it in `toks`. Run:
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/airristotle/test_prompt.py tests/alignair/airristotle/test_tokenizer.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/airristotle/prompt.py src/alignair/airristotle/tokenizer.py tests/alignair/airristotle/test_prompt.py
git commit -m "airristotle: prompt-builder with copy-pointer targets (calls+coords copy into prompt)"
```

---

### Task 3: Config + decoder-only backbone (RMSNorm/RoPE/SwiGLU/GQA, causal)

**Files:**
- Create: `src/alignair/airristotle/config.py`, `src/alignair/airristotle/model.py`
- Test: `tests/alignair/airristotle/test_model.py`

**Interfaces:**
- Produces: `@dataclass AIRRConfig(vocab_size:int, d_model:int=768, n_layers:int=12, n_heads:int=12, n_kv_heads:int=4, d_ff:int=2048, max_seq:int=8192, rope_base:float=10000.0)`; `AIRRistotle(cfg)` with `.forward(input_ids: LongTensor[B,L]) -> (hidden: [B,L,d_model], lm_logits: [B,L,vocab])`, causal. `.n_params() -> int`.
- Reuse: the RoPE approach from `src/alignair/nn/encoder/shared.py` (`_apply_rope`), but implement a CAUSAL, GQA attention + RMSNorm here (shared.py is bidirectional MHA).

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/airristotle/test_model.py
import torch
from alignair.airristotle.config import AIRRConfig
from alignair.airristotle.model import AIRRistotle


def test_forward_shapes_and_causal():
    cfg = AIRRConfig(vocab_size=50, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=256, max_seq=64)
    m = AIRRistotle(cfg).eval()
    ids = torch.randint(0, 50, (2, 16))
    hid, logits = m(ids)
    assert hid.shape == (2, 16, 128)
    assert logits.shape == (2, 16, 50)
    # causal: changing a LATER token must not change an EARLIER position's logits
    ids2 = ids.clone(); ids2[:, -1] = (ids2[:, -1] + 1) % 50
    _, l2 = m(ids2)
    assert torch.allclose(logits[:, 0], l2[:, 0], atol=1e-5)


def test_150m_config_param_count_in_range():
    cfg = AIRRConfig(vocab_size=50)                 # defaults ~150M
    m = AIRRistotle(cfg)
    n = m.n_params()
    assert 120_000_000 <= n <= 190_000_000, n
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/airristotle/test_model.py -q`
Expected: FAIL (missing modules).

- [ ] **Step 3: Implement**

```python
# src/alignair/airristotle/config.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class AIRRConfig:
    vocab_size: int
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 2048
    max_seq: int = 8192
    rope_base: float = 10000.0
```

```python
# src/alignair/airristotle/model.py
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
    f = torch.outer(t, inv)                                  # (L, hd/2)
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
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # causal mask
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

    def forward(self, input_ids):
        B, L = input_ids.shape
        x = self.emb(input_ids)
        cos, sin = _rope(L, self.hd, self.cfg.rope_base, input_ids.device)
        cos, sin = cos[None, None], sin[None, None]
        for b in self.blocks:
            x = b(x, cos, sin)
        x = self.norm(x)
        return x, self.lm_head(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/airristotle/test_model.py -q`
Expected: PASS. (If `n_params` is out of [120M,190M], adjust defaults — e.g. d_model 768/n_layers 12 with vocab ~50 gives ~110M from blocks; bump `n_layers` to 14 or `d_ff` to 2304 to land in range. Blocks dominate; embedding is tiny with a ~50-token vocab.)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/airristotle/config.py src/alignair/airristotle/model.py tests/alignair/airristotle/test_model.py
git commit -m "airristotle: decoder-only backbone (RMSNorm/RoPE/SwiGLU/causal-GQA), ~150M config"
```

---

### Task 4: Copy head + two-channel loss

**Files:**
- Modify: `src/alignair/airristotle/model.py`
- Test: `tests/alignair/airristotle/test_loss.py`

**Interfaces:**
- Consumes: `AIRRistotle.forward` returning `(hidden[B,L,d], lm_logits[B,L,vocab])` (Task 3); an `Example`'s target arrays (Task 2).
- Produces: on the model, `.copy_logits(hidden, prompt_len:int) -> [B,L,prompt_len]` (attention scores from each decode position to the PROMPT positions). And a module-level
  `airristotle_loss(lm_logits, copy_logits, batch) -> Tensor` where `batch` has padded tensors
  `gen_target[B,L], copy_target[B,L], is_copy[B,L], loss_mask[B,L]` and labels are for position t
  (already aligned: position t's row predicts label[t]); loss = masked CE over lm_logits where
  `is_copy==0` + masked CE over copy_logits where `is_copy==1`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/airristotle/test_loss.py
import torch
from alignair.airristotle.config import AIRRConfig
from alignair.airristotle.model import AIRRistotle, airristotle_loss


def test_copy_logits_shape_and_loss_runs():
    cfg = AIRRConfig(vocab_size=50, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=128, max_seq=64)
    m = AIRRistotle(cfg)
    ids = torch.randint(0, 50, (2, 20)); prompt_len = 12
    hid, lm = m(ids)
    cp = m.copy_logits(hid, prompt_len)
    assert cp.shape == (2, 20, prompt_len)
    batch = {"gen_target": torch.randint(0, 50, (2, 20)),
             "copy_target": torch.randint(0, prompt_len, (2, 20)),
             "is_copy": torch.randint(0, 2, (2, 20)),
             "loss_mask": torch.ones(2, 20, dtype=torch.long)}
    loss = airristotle_loss(lm, cp, batch)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_perfect_copy_gives_low_copy_loss():
    # Loss uses the causal next-token SHIFT: hidden[t] predicts label[t+1]. So a label at position 1
    # is predicted from copy_logits[:, 0]. Point position-0's copy hard at prompt pos 2 -> copy term ~0.
    cfg = AIRRConfig(vocab_size=10, d_model=32, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=64, max_seq=32)
    m = AIRRistotle(cfg)
    lm = torch.zeros(1, 3, 10)
    cp = torch.full((1, 3, 4), -10.0); cp[0, 0, 2] = 10.0     # hidden0 points hard at prompt pos 2
    batch = {"gen_target": torch.zeros(1, 3, dtype=torch.long),
             "copy_target": torch.tensor([[0, 2, 0]]),        # label at position 1 = copy pos 2
             "is_copy": torch.tensor([[0, 1, 0]]),            # position 1 is a copy label
             "loss_mask": torch.tensor([[0, 1, 0]])}          # only position 1 contributes
    assert airristotle_loss(lm, cp, batch).item() < 0.01
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/airristotle/test_loss.py -q`
Expected: FAIL (`copy_logits`/`airristotle_loss` missing).

- [ ] **Step 3: Implement (append to model.py)**

```python
# append to src/alignair/airristotle/model.py

# copy head: a learned projection of the decode-position hidden state, scored (dot-product) against
# the PROMPT positions' hidden states — an attention distribution over "which prompt token to copy".
def _add_copy_head(self):
    self.copy_q = nn.Linear(self.cfg.d_model, self.cfg.d_model, bias=False)
    self.copy_k = nn.Linear(self.cfg.d_model, self.cfg.d_model, bias=False)

AIRRistotle._add_copy_head = _add_copy_head

# call _add_copy_head in __init__: (edit __init__ to append `self._add_copy_head()` at the end)

def copy_logits(self, hidden, prompt_len):
    q = self.copy_q(hidden)                                  # (B,L,d) queries from every decode pos
    k = self.copy_k(hidden[:, :prompt_len])                 # (B,P,d) keys from prompt positions
    scale = q.shape[-1] ** -0.5
    return torch.einsum("bld,bpd->blp", q, k) * scale        # (B,L,P)

AIRRistotle.copy_logits = copy_logits


def airristotle_loss(lm_logits, copy_logits, batch):
    # Causal next-token shift: hidden[t] (logits[:, t]) predicts the label at position t+1. Predicting a
    # position's own token would be leaky/degenerate, so we always shift by one.
    lm = lm_logits[:, :-1]                                    # (B, L-1, V)
    cp = copy_logits[:, :-1]                                  # (B, L-1, P)
    P = cp.shape[-1]
    gen_t = batch["gen_target"][:, 1:]
    copy_t = batch["copy_target"][:, 1:].clamp(max=P - 1)
    mask = batch["loss_mask"][:, 1:].bool()
    is_copy = batch["is_copy"][:, 1:].bool() & mask
    is_gen = (~batch["is_copy"][:, 1:].bool()) & mask
    total = lm_logits.new_zeros(())
    n = mask.sum().clamp(min=1)
    if is_gen.any():
        total = total + F.cross_entropy(lm[is_gen], gen_t[is_gen], reduction="sum")
    if is_copy.any():
        total = total + F.cross_entropy(cp[is_copy], copy_t[is_copy], reduction="sum")
    return total / n
```

Then edit `AIRRistotle.__init__` to call `self._add_copy_head()` on the last line, and ensure
`airristotle_loss` is importable at module level (it is — it's a top-level `def`).

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/airristotle/test_loss.py tests/alignair/airristotle/test_model.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/airristotle/model.py tests/alignair/airristotle/test_loss.py
git commit -m "airristotle: copy head over prompt positions + two-channel (generate|copy) loss"
```

---

### Task 5: Batching + training loop + overfit sanity (the MVP verdict)

**Files:**
- Create: `scripts/train_airristotle.py`, `src/alignair/airristotle/data.py`
- Test: `tests/alignair/airristotle/test_overfit.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `src/alignair/airristotle/data.py` with `collate(examples, pad_id:int) -> dict[str,Tensor]` (pads `input_ids/gen_target/copy_target/is_copy/loss_mask` to max len; also returns per-example `prompt_len` list) and `stream_examples(reference_set, tokenizer, params:dict, n:int, seed:int, n_distractors:int)` yielding `Example`s from the gym. `scripts/train_airristotle.py` = an argparse training loop (bf16, AdamW, cosine, grad-clip).

- [ ] **Step 1: Write the failing test (overfit = MVP success criterion)**

```python
# tests/alignair/airristotle/test_overfit.py
import pytest, torch
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.airristotle.tokenizer import AIRRTokenizer
from alignair.airristotle.config import AIRRConfig
from alignair.airristotle.model import AIRRistotle, airristotle_loss
from alignair.airristotle.prompt import build_example
from alignair.airristotle.data import collate, stream_examples
from alignair.reference.reference_set import ReferenceSet


def test_overfits_tiny_set():
    """The MVP bet: a small model can LEARN to copy calls+coords out of the prompt. Overfit 8 fixed
    examples; the combined loss must fall well below its initial value."""
    torch.manual_seed(0)
    tok = AIRRTokenizer(); rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    params = dict(mutation_rate=0.0, productive_only=False, end_loss_5=(0, 0), end_loss_3=(0, 0),
                  indel_count=(0, 0), seq_error_rate=0.0, ambiguous_count=(0, 0))
    exs = list(stream_examples(rs, tok, params, n=8, seed=1, n_distractors=4))
    batch = collate(exs, pad_id=tok.id(tok.PAD))
    cfg = AIRRConfig(vocab_size=tok.vocab_size, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
                     d_ff=256, max_seq=batch["input_ids"].shape[1])
    m = AIRRistotle(cfg).train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    P = int(batch["prompt_len"].min())
    def step():
        hid, lm = m(batch["input_ids"]); cp = m.copy_logits(hid, hid.shape[1])
        return airristotle_loss(lm, cp, batch)
    l0 = step().item()
    for _ in range(60):
        opt.zero_grad(); loss = step(); loss.backward(); opt.step()
    l1 = step().item()
    assert l1 < 0.5 * l0, (l0, l1)                 # learned to copy/generate the fixed set
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/airristotle/test_overfit.py -q`
Expected: FAIL (`data.py` missing).

- [ ] **Step 3: Implement data.py**

```python
# src/alignair/airristotle/data.py
"""Batching + a gym-backed example stream for AIRRistotle training."""
from __future__ import annotations
import torch
from .prompt import build_example


def stream_examples(reference_set, tokenizer, params, n, seed, n_distractors=8):
    import random
    from ..gym.gym import build_experiment
    rng = random.Random(seed)
    exp = build_experiment(_dataconfig_of(reference_set), params)
    for rec in exp.stream_records(n=n, seed=seed):
        yield build_example(rec, reference_set, tokenizer, n_distractors=n_distractors, rng=rng)


def _dataconfig_of(reference_set):
    import GenAIRR.data as gdata
    return gdata.HUMAN_IGH_OGRDB                   # MVP: IGH only


def collate(examples, pad_id):
    L = max(len(e.input_ids) for e in examples)
    def pad(seq, val): return seq + [val] * (L - len(seq))
    def T(key, val, dtype=torch.long):
        return torch.tensor([pad(getattr(e, key), val) for e in examples], dtype=dtype)
    return {
        "input_ids": T("input_ids", pad_id),
        "gen_target": T("gen_target", 0),
        "copy_target": T("copy_target", 0),
        "is_copy": T("is_copy", 0),
        "loss_mask": T("loss_mask", 0),
        "prompt_len": torch.tensor([e.prompt_len for e in examples], dtype=torch.long),
    }
```

- [ ] **Step 4: Run the overfit test**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/airristotle/test_overfit.py -q`
Expected: PASS (final loss < half the initial). If it does not drop, that is a real signal — inspect the prompt-target alignment (Task 2) before increasing steps.

- [ ] **Step 5: Implement the training script + commit**

```python
# scripts/train_airristotle.py
"""Train AIRRistotle on the GenAIRR gym (MVP: small genotype, forward clean/moderate reads)."""
import argparse, math, torch
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.airristotle.tokenizer import AIRRTokenizer
from alignair.airristotle.config import AIRRConfig
from alignair.airristotle.model import AIRRistotle, airristotle_loss
from alignair.airristotle.data import collate, stream_examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-distractors", type=int, default=8)
    ap.add_argument("--progress", type=float, default=0.3)
    ap.add_argument("--out", default=".private/models/airristotle_mvp.pt")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AIRRTokenizer(); rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    from alignair.gym.curriculum import Curriculum
    params = dict(Curriculum().params(a.progress))
    cfg = AIRRConfig(vocab_size=tok.vocab_size)
    m = AIRRistotle(cfg).to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    gen = stream_examples(rs, tok, params, n=a.steps * a.batch_size, seed=0, n_distractors=a.n_distractors)
    buf = []
    for step in range(a.steps):
        exs = [next(gen) for _ in range(a.batch_size)]
        batch = {k: v.to(dev) for k, v in collate(exs, tok.id(tok.PAD)).items()}
        lr = a.lr * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        for pg in opt.param_groups: pg["lr"] = lr
        with torch.autocast(dev, dtype=torch.bfloat16):
            hid, lm = m(batch["input_ids"]); cp = m.copy_logits(hid, hid.shape[1])
            loss = airristotle_loss(lm, cp, batch)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 50 == 0:
            print(f"step {step}  loss {loss.item():.4f}  lr {lr:.2e}", flush=True)
    torch.save({"model": m.state_dict(), "config": vars(cfg)}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
```

```bash
git add scripts/train_airristotle.py src/alignair/airristotle/data.py tests/alignair/airristotle/test_overfit.py
git commit -m "airristotle: gym stream + collate + training loop + overfit sanity (MVP verdict)"
```

---

## Done criteria (MVP)

- `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/airristotle/ -q` all green — including the **overfit test** (the model *can* learn to copy calls + coordinates out of the prompt).
- A short real run: `PYTHONPATH=src .venv/bin/python scripts/train_airristotle.py --steps 500 --batch-size 8` shows the loss trending down (not a formal eval yet — that's the next plan).
- **Verdict:** if the model overfits and a short run's loss falls steadily, the core hypothesis (an LLM can do IG alignment by pointing) holds → proceed to the next plan (proper decode + benchmark eval, then dynamic genotype, then RLVR). If it can't overfit, stop and debug the prompt/target alignment before scaling.

## Follow-on plans (not this plan)
1. **Decode + benchmark adapter** — greedy/beam decode of the record, `airristotle_predictor`, score on the benchmark vs XAttnAligner/IgBLAST (clean/moderate first).
2. **Dynamic genotype** — genotype-subset sampling + synthetic novel-allele injection + held-out alleles; verify on the dynamic-genotype strata.
3. **Full difficulty + RLVR/GRPO** against the benchmark metric.
4. **Scale + speed** — grow toward ~1B if capacity-bound; non-autoregressive decoding.
