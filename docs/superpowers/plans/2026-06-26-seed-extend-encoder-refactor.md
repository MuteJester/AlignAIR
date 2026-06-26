# Seed-and-Extend: Shared-Encoder Refactor + Gate-1 Repeat + Neural-Contribution Ablation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Collapse the read and reference encoders into ONE shared type-embedded encoder (deleting the separate `GermlineEncoder` double-encode and the dynamic-genotype-violating `caller="classifier"` path), retrain from scratch, repeat Gate-1 on the co-trained encoder, and run the neural-contribution ablation that defends "deep neural aligner with an exact differentiable DP decoder, not a repackaged classical aligner."

**Architecture:** `backbone="shared"` (`SharedNucleotideEncoder`) already has a read/germline `type_emb` + `forward_positions(tokens, mask, token_type)`. The refactor encodes germline references and segments through THAT encoder (token_type=GERMLINE), reads segment reps OFF the backbone (no re-encode → kills the profiled 14% re-encode), caches reference reps and refreshes them during training, and exposes config toggles so the neural-contribution ablation is a set of config flips.

**Tech Stack:** PyTorch (`.venv/bin/python`, `PYTHONPATH=src`), pytest, the GenAIRR gym + FrozenLattice, the seed-and-extend modules in `nn/aligner/`.

**Spec:** `docs/superpowers/specs/2026-06-25-structural-seed-and-extend-neural-dp-design.md` (build step 2 + the Gate-1 repeat [build step 4] + the §5.1 neural-contribution ablation, now mandatory for Gate 3).

## Global Constraints

- Run with `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python`. `alignair` is NOT pip-installed. NEVER bare `python`.
- Git messages: NEVER add Co-Authored-By / Claude mentions (project rule).
- **Dynamic genotype (hard rule):** references are runtime inputs, encoded + cached; novel alleles callable; NO allele identity in weights → the `caller="classifier"` path is REMOVED from the compliant path.
- **Segmentation-first:** retrieval / band / DP all operate on the segmentation-gated V/D/J segment.
- The refactor REQUIRES `backbone="shared"` (the conv backbone has no germline type-embedding / `forward_positions`). Keep the conv path working but mark shared as the only seed-extend-compatible backbone.
- Keep `nn/aligner/soft_dp.py` (`SoftDPAligner`) as the A/B oracle and `nn/aligner/banded_dp.py` (`SeedExtendAligner`) from build step 3 unchanged.
- TDD for code rewiring; experiment-style (run + record + decide) for retrains/gates/ablations.

## File Structure

- `src/alignair/config/dnalignair_config.py` — MODIFY: add ablation toggles (one block, all default to the full-neural config).
- `src/alignair/core/dnalignair.py` — MODIFY: route germline + segment encoding through the shared encoder; delete `GermlineEncoder` instantiation + the classifier path; read segment reps off backbone reps.
- `src/alignair/training/germline_tf.py` — MODIFY: stop re-encoding the segment via `germline_encoder`; consume backbone segment reps.
- `src/alignair/inference/dnalignair_infer.py` — MODIFY: same de-double-encode in the rerank path.
- `scripts/exp_neural_contribution_ablation.py` — NEW: the §5.1 ablation suite (config flips → headline metrics).
- Tests: `tests/alignair/core/test_shared_encoder_refactor.py`.

## Ablation toggles (config) — the full-neural defaults, each flippable for §5.1

Add to `DNAlignAIRConfig` (all default to the full-neural system; the ablation script flips them):
- `band_features: str = "full"`        # "full" (base-match + k-mer + boundary + learned cosine) | "raw" (base-match/k-mer only)
- `dp_emissions: str = "learned"`      # "learned" (token reps + projections + scale) | "raw" (raw +1/-1 base-match only)
- `use_learned_reps: bool = True`      # False → emission/band use raw base-match channel only
- `use_reliability: bool = True`       # state-conditioned SHM reliability into the DP emission
- `reader: str = "dp"`                 # "dp" (log-partition, rule 1) | "pooled" | "maxsim"
- `encoder_mode: str = "trained"`      # "trained" | "frozen" | "random" (ablation: is the learned encoder load-bearing?)

These already partly exist conceptually (the DP emission's learned projections + scale + gaps, the reliability channel). The plan WIRES them to config so the ablation is config-driven, not code-forking.

---

## Task 1: Config — shared-encoder mode + ablation toggles

**Files:**
- Modify: `src/alignair/config/dnalignair_config.py`
- Test: `tests/alignair/core/test_shared_encoder_refactor.py`

**Interfaces:**
- Produces: `DNAlignAIRConfig` with the six ablation fields above + an `aligner="seed_extend"` value.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/core/test_shared_encoder_refactor.py
from alignair.config.dnalignair_config import DNAlignAIRConfig


def test_config_has_seed_extend_and_ablation_toggles():
    c = DNAlignAIRConfig(aligner="seed_extend", backbone="shared")
    assert c.aligner == "seed_extend"
    for f, default in [("band_features", "full"), ("dp_emissions", "learned"),
                       ("use_learned_reps", True), ("use_reliability", True),
                       ("reader", "dp"), ("encoder_mode", "trained")]:
        assert getattr(c, f) == default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/core/test_shared_encoder_refactor.py -q`
Expected: FAIL — `unexpected keyword argument` / missing attribute.

- [ ] **Step 3: Implement** — add to the `DNAlignAIRConfig` dataclass:

```python
    aligner: str = "softdp"  # ... | "pointer" | "seed_extend" (shared encoder + banded DP)
    band_features: str = "full"     # "full" | "raw"
    dp_emissions: str = "learned"   # "learned" | "raw"
    use_learned_reps: bool = True
    use_reliability: bool = True
    reader: str = "dp"              # "dp" | "pooled" | "maxsim"
    encoder_mode: str = "trained"   # "trained" | "frozen" | "random"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/core/test_shared_encoder_refactor.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/config/dnalignair_config.py tests/alignair/core/test_shared_encoder_refactor.py
git commit -m "Add seed_extend aligner + neural-contribution ablation toggles to config"
```

---

## Task 2: Route germline encoding through the shared encoder; delete GermlineEncoder + classifier

**Files:**
- Modify: `src/alignair/core/dnalignair.py` (`__init__`, `encode_reference`, `match_alleles`)
- Test: `tests/alignair/core/test_shared_encoder_refactor.py`

**Interfaces:**
- Consumes: `SharedNucleotideEncoder.forward_positions(tokens, mask, token_type)` and its `GERMLINE`/`READ` constants (`nn/encoder/shared.py:104`).
- Produces: a `DNAlignAIR` with NO `self.germline_encoder` and NO `self.classifier` when `aligner="seed_extend"`; `encode_reference` uses the shared backbone with `token_type=GERMLINE`; segment reps for retrieval are READ OFF the backbone reps (no re-encode).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/alignair/core/test_shared_encoder_refactor.py
import torch
from alignair.core.dnalignair import DNAlignAIR


def test_seed_extend_has_no_germline_encoder_or_classifier():
    cfg = DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, backbone="shared",
                           aligner="seed_extend")
    m = DNAlignAIR(cfg)
    assert not hasattr(m, "germline_encoder") or m.germline_encoder is None
    assert not hasattr(m, "classifier")


def test_encode_reference_uses_shared_encoder():
    cfg = DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, backbone="shared",
                           aligner="seed_extend")
    m = DNAlignAIR(cfg)
    # a tiny reference-set stub with one gene's sequences
    class _Gene:
        sequences = ["ACGTACGTAC", "ACGTTCGTAC"]
        names = ["IGHVx*01", "IGHVx*02"]
    class _RS:
        genes = {"V": _Gene()}
        has_d = False
        def gene(self, g): return _Gene()
    ref = m.encode_reference(_RS())
    assert ref["V"]["pos_reps"].shape[0] == 2          # two alleles encoded
    assert ref["V"]["pos_reps"].shape[-1] == 32        # d_model
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/core/test_shared_encoder_refactor.py -q`
Expected: FAIL — `germline_encoder` still present / classifier present.

- [ ] **Step 3: Implement** — in `DNAlignAIR.__init__`, gate the germline encoder + classifier on the aligner:

```python
        self.seed_extend = getattr(config, "aligner", "diagonal") == "seed_extend"
        # The shared encoder embeds BOTH read and reference (type_emb). For seed_extend there is
        # NO separate GermlineEncoder (kills the double-encode) and NO classifier (dynamic-genotype).
        if not self.seed_extend:
            self.germline_encoder = GermlineEncoder(embed_dim=d)
        self.caller = getattr(config, "caller", "retrieval")
        if self.caller == "classifier" and not self.seed_extend:
            counts = config.allele_counts or {}
            self.classifier = nn.ModuleDict({g: nn.Linear(d, counts[g]) for g in counts})
```

Add a `_germ_encode(tok, msk)` helper that uses the shared backbone for seed_extend, else the
germline encoder:

```python
    def _germ_encode_positions(self, tok, msk):
        if self.seed_extend:
            from ..nn.encoder.shared import GERMLINE
            return self.backbone.forward_positions(tok, msk, token_type=GERMLINE)
        return self.germline_encoder.forward_positions(tok, msk)

    def _germ_encode_pooled(self, tok, msk):
        if self.seed_extend:
            from ..nn.encoder.shared import GERMLINE
            return self.backbone(tok, msk, token_type=GERMLINE)
        return self.germline_encoder(tok, msk)
```

Update `encode_reference` to use `self._germ_encode_pooled` / `self._germ_encode_positions`. Update
`match_alleles`: for seed_extend, the retrieval query reps are READ OFF the backbone reps (the
already-encoded `reps` for the segment positions) — gather the segment reps via `extract_segment`
instead of re-encoding the segment tokens.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/core/test_shared_encoder_refactor.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/core/dnalignair.py tests/alignair/core/test_shared_encoder_refactor.py
git commit -m "Route germline+segment encoding through the shared encoder; drop GermlineEncoder+classifier for seed_extend"
```

---

## Task 3: De-double-encode the training + inference coord paths

**Files:**
- Modify: `src/alignair/training/germline_tf.py`, `src/alignair/inference/dnalignair_infer.py`

**Interfaces:**
- Consumes: backbone segment reps (no re-encode); `_germ_encode_positions` for the reference.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/alignair/core/test_shared_encoder_refactor.py
def test_compute_germline_logits_no_reencode_for_seed_extend(monkeypatch):
    # seed_extend must NOT call a separate germline_encoder.forward_positions on the segment
    cfg = DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, backbone="shared", aligner="seed_extend")
    m = DNAlignAIR(cfg)
    assert not hasattr(m, "germline_encoder") or m.germline_encoder is None
    # (smoke: a 1-batch teacher-forced call runs end to end without a germline_encoder)
```

- [ ] **Step 2: Run / Step 3: Implement** — in `compute_germline_logits` (germline_tf.py) and the
inference rerank (dnalignair_infer.py:299-300), replace `model.germline_encoder.forward_positions(seg_tok, seg_mask)`
with the backbone segment reps gathered from the already-encoded read reps (pass `reps` through and
`extract_segment(reps, mask, region_labels, gene)`), and `ref_emb[...]["pos_reps"]` continues to hold
the shared-encoder germline reps. Reference encoding uses `model._germ_encode_positions`.

- [ ] **Step 4: Run the training + inference suites**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/training/ tests/alignair/inference/ tests/alignair/integration/ -q`
Expected: PASS (the non-seed_extend path is unchanged; seed_extend uses backbone reps).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/training/germline_tf.py src/alignair/inference/dnalignair_infer.py tests/alignair/core/test_shared_encoder_refactor.py
git commit -m "Consume backbone segment reps in coord paths (kill the 14% re-encode) for seed_extend"
```

---

## Task 4: Retrain from scratch + verify retrieval recall@k and coord parity

**Files:**
- Reuse: `scripts/exp_aligner_ablation.py` pattern / `GymTrainer`; no new module.

**Interfaces:**
- Consumes: the refactored model with `aligner="seed_extend"`, the band head + `SeedExtendAligner`.

- [ ] **Step 1: Train** a `seed_extend` model from scratch on the gym (IGH, the production-ish config), with the band head co-trained (band offset loss added to the trainer's loss). Record steps + wall time.

- [ ] **Step 2: Verify** on the frozen lattice: retrieval recall@k (true allele ∈ top-k) and coordinate competence vs the soft-DP oracle — must be ≥ the oracle's (bootstrap-CI lower bound) on clean/heavy-SHM/indel. If retrieval recall or coord parity REGRESSES vs re-encoding, fall back to encoding the segment via the shared encoder with `token_type=GERMLINE` (keeps shared weights, pays the re-encode) and re-verify — i.e. the "no re-encode" is the target, the shared-weight re-encode is the safe fallback.

- [ ] **Step 3: Record** results + decision to memory; commit any trainer wiring.

---

## Task 5: Gate-1 REPEAT on the co-trained shared encoder

**Files:**
- Reuse: `scripts/exp_band_recall_gate.py` pointed at the refactored model.

**Interfaces:**
- Consumes: the Task-4 retrained `seed_extend` model.

- [ ] **Step 1: Run** the band-recall gate (top-m union recall + committed-recall + fail-open + cell budget, per cell, w=8/16) on the CO-TRAINED encoder (not the frozen 8h proxy).

- [ ] **Step 2: Compare** vs the frozen-proxy Gate 1 (clean/heavy-SHM/indel = 1.0; junction 0.97). The co-trained encoder may LIFT junction past 0.97 (the spec's stated reason for the repeat). Record the per-cell table + whether junction improves. Decision: junction committed-recall should be ≥ the frozen-proxy 0.97; if it regresses, debug before the kernel.

---

## Task 6: Neural-contribution ablation suite (§5.1 — the Gate-3 defense)

**Files:**
- Create: `scripts/exp_neural_contribution_ablation.py`

**Interfaces:**
- Consumes: the refactored model + the config toggles (Task 1); the frozen lattice; the dry-run metrics.

- [ ] **Step 1: Write the ablation runner** — for each toggle config, measure the HEADLINE metrics
(heavy-SHM-V allele accuracy via the DP rerank + frozen-lattice coord competence), with bootstrap CIs:

```python
# scripts/exp_neural_contribution_ablation.py  (sketch — mirrors exp_seed_extend_gate2_dryrun.py)
# Configs to compare (all on the SAME trained weights where the toggle is inference-time,
# separate trains only where the toggle is structural: encoder_mode, use_learned_reps):
#   full         : band_features=full, dp_emissions=learned, use_learned_reps=True,
#                  use_reliability=True, reader=dp, encoder_mode=trained   (the system)
#   raw_band     : band_features=raw
#   raw_dp       : dp_emissions=raw            (raw +1/-1 emission only)
#   parasail_only: reader=pooled + classical parasail coords (io/alignment.py) — the "is it IgBLAST?" probe
#   no_reps      : use_learned_reps=False
#   no_reliab    : use_reliability=False
#   maxsim_reader: reader=maxsim
#   pooled_reader: reader=pooled
#   frozen_enc   : encoder_mode=frozen   (random/frozen encoder, retrained heads)
# Report: heavy-SHM-V allele acc + clean/heavy_shm/indel/junction coord competence per config.
```

- [ ] **Step 2: Run** the ablation on the Task-4 model.

- [ ] **Step 3: Pass condition (spec §5.1):** the FULL system beats `raw_band` / `raw_dp` /
`parasail_only` by a material, CI-disjoint margin on heavy-SHM-V (the dry run already shows the learned
DP reader moves heavy-SHM-V 0.53→0.91 on the current encoder). If `parasail_only` MATCHES `full`, the
model collapsed to classical alignment — STOP and rethink. Otherwise the neural-contribution claim is
defended: record the table to memory as the paper-ready evidence.

- [ ] **Step 4: Commit**

```bash
git add scripts/exp_neural_contribution_ablation.py
git commit -m "Add neural-contribution ablation suite (defends learned DP decoder vs classical alignment)"
```

---

## Notes on scope / what comes next

This plan delivers the encoder refactor (build step 2), the co-trained Gate-1 repeat (build step 4),
and the §5.1 neural-contribution ablation. It does NOT build the fused Triton kernel (build step 5) —
that follows, validated vs the sequential `SeedExtendAligner` reference, and only matters for SPEED
(the accuracy story is already carried by the sequential DP). After this plan: fused kernel → full
Gate 2 (integrated, refactored encoder) → Gate 3 (A/B vs soft-DP oracle: competence + coord parity +
speed at B=64, WITH the neural-contribution ablation as the defense).
