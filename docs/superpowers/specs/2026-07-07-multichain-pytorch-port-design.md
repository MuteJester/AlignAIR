# MultiChain AlignAIR — PyTorch Port Design

**Date:** 2026-07-07
**Goal:** Faithfully port the TF `MultiChainAlignAIR` to clean PyTorch, matching the old TF logic, reusing the already-ported SingleChain backbone. No training — logic port only.

## Background

The TF `MultiChainAlignAIR` (`src/AlignAIR/Models/MultiChainAlignAIR/MultiChainAlignAIR.py` on `main`) is
byte-for-byte the same architecture as `SingleChainAlignAIR` **except** it adds a single new task:
predicting the **chain type** (locus) of each read. A direct diff of the two TF files confirms the
only functional deltas are:

1. Config source is a `MultiDataConfigContainer` (multiple dataconfigs) rather than one `DataConfig`;
   allele counts and `has_d` become the **union** across chains.
2. `number_of_types = len(dataconfigs.chain_types())`.
3. A `chain_type` head off the **meta** tower: `Dense(max_seq_length, gelu) -> Dropout(0.05) ->
   Dense(number_of_types, softmax)`.
4. A Kendall log-var `log_var_chain_type`.
5. A categorical-cross-entropy `chain_type` loss term, Kendall-weighted, added to the total loss.

Everything else (embedding, meta/seg/cls feature towers, segmentation heads, soft-cutout masking,
allele + mutation/indel/productivity heads, and the entire hierarchical loss) is identical to
SingleChain, which is already faithfully ported in `src/alignair/models/`.

Our infra already supports the multi-chain reference: `ReferenceSet.from_dataconfigs(*dataconfigs)`
accepts multiple dataconfigs and unions genes with `has_d = any(...)`. The AIRR builder already
reads a `locus` field (`rec.get("locus", "IGH")`).

## Design

Approach: **subclass** `MultiChainAlignAIR(SingleChainAlignAIR)`. A real MultiChain class (matching TF's
two-class identity, so predict / serialization can branch on model type) that reuses 100% of the
backbone and adds only the chain_type head. This avoids TF's copy-paste and keeps the tree clean.

Seam: `SingleChainAlignAIR.forward` gains one call `self._extra_meta_heads(meta, out)` after the
analysis heads; the base implementation is a no-op, so SingleChain behavior is byte-identical.
MultiChain overrides it to compute `out["chain_type_logits"]`.

The chain_type **loss** term is added *conditionally* in `hierarchical_loss` — active only when both
`out["chain_type_logits"]` and `targets["chain_type"]` are present — exactly mirroring the existing
conditional `orientation` term. `make_logvars` includes a `"chain_type"` weight when
`cfg.num_chain_types > 1`.

Consistent with our orientation head, chain_type is stored/trained as **logits** with
`F.cross_entropy` (numerically stable) rather than TF's softmax-probs + `CategoricalCrossentropy`;
the two are equivalent for one-hot / class-index targets.

### Predict → locus

- `clean()` adds `chain_type` (argmax of `chain_type_logits`) to `Predictions` when present.
- `PredictConfig` gains `chain_types: Optional[tuple[str, ...]]` (ordered names matching training).
- `pipeline._to_records` sets `rec["locus"] = cfg.chain_types[idx]` when both are available; the AIRR
  builder already consumes `locus`.

## File changes

- `config/alignair_config.py` — add `num_chain_types: int = 1`.
- `models/single_chain.py` — add `_extra_meta_heads(meta, out)` no-op + call it in `forward`.
- `models/multi_chain.py` *(new)* — `MultiChainAlignAIR(SingleChainAlignAIR)` with the three
  chain_type layers + `_extra_meta_heads` override.
- `models/losses.py` — `make_logvars` adds `"chain_type"` (multi-chain only); `hierarchical_loss`
  adds a conditional chain_type CE term.
- `models/__init__.py` — export `SingleChainAlignAIR`, `MultiChainAlignAIR`.
- `predict/state.py` — `Predictions.chain_type: Optional[np.ndarray]`.
- `predict/clean.py` — populate `chain_type`.
- `predict/config.py` — `PredictConfig.chain_types`.
- `predict/pipeline.py` — map chain_type index -> `rec["locus"]`.
- `tests/alignair/models/test_multi_chain.py` *(new)* — chain_type logits shape, gradient flow, loss
  term present, has_d / light-chain variants, SingleChain-emits-no-chain_type parity.
- `tests/alignair/predict/test_multichain_locus.py` *(new)* — index -> locus mapping.

## Non-goals

Training, multi-chain data streaming / targets in the trainer, and serialization-bundle parity are
out of scope for this slice (logic port only, per the request).
