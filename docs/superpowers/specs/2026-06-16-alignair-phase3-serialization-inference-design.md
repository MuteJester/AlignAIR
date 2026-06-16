# AlignAIR PyTorch Migration — Phase 3 (Serialization + Inference) Design

**Date:** 2026-06-16
**Status:** 3a + 3b COMPLETE. **3c (germline alignment + full AIRR output) DEFERRED to the
model-architecture-redesign phase** — see "Deferral" below.
**Depends on:** Phase 1 (DL core) + Phase 2 (data + training) — complete on `pytorch-migration`
**Parent design:** `docs/superpowers/specs/2026-06-15-alignair-pytorch-migration-design.md`

## Deferral (2026-06-16): drop the heuristic germline matcher; predict the reference alignment instead

Decision: do **not** port `HeuristicReferenceMatcher` / the legacy germline-alignment stage.
Its job — recovering per-segment **reference** coordinates (`start_in_ref`/`end_in_ref`, i.e. the
5'/3' trims) by aligning each predicted segment to the germline — will instead be handled by the
**model itself** during the architecture redesign: add heads that predict the trims / germline
start-end directly. Those targets are already labels in the training data (CSV: `*_germline_start/end`,
`*_trim_5/3`; GenAIRR `stream_records`: same fields), so they are trainable now.

Consequences:
- Phase 3 ships at **3a (serialization) + 3b (inference core: calls + padding-corrected in-sequence
  coordinates)**. That is the migration-complete inference path.
- A thin **reference-gapping + AIRR-table** layer (deterministic assembly from
  `allele + germline_start/end`, needs the `DataConfig` references) is deferred and built alongside the
  new germline/trim heads in the architecture-redesign phase.
- Boundary *refinement* the matcher did (snapping a slightly-off boundary to the reference) is dropped;
  it relies on the trained boundary heads being accurate.
- Junction/CDR/FWR regions, `productive`, `vj_in_frame`, `stop_codon` are computable from in-sequence
  coordinates without germline alignment if a lite-AIRR output is wanted before the redesign.

The remainder of this document (the original 3b/3c plan) is retained for reference but §4's germline +
AIRR portions are superseded by the deferral above.

## 1. Goal

Replace the TensorFlow SavedModel serialization/inference path with a clean
PyTorch one: a `state_dict`-based model bundle and an end-to-end inference
pipeline that loads a saved model and predicts on CSV/FASTA input, producing
allele calls, corrected coordinates, and a full AIRR-format output table.

### Decisions (confirmed with user)

| Decision | Choice |
| --- | --- |
| Inference depth | **Full inference incl. AIRR output** — predictor + decode + allele thresholding + segment correction + germline alignment + AIRR table builder + CSV/AIRR serialization. |
| Orientation | **Deferred.** Orientation detection/correction will be integrated into the model architecture during the later model-architecture-upgrade work. Phase 3 assumes correctly-oriented input. |
| Decomposition | **Two ordered sub-plans: 3a (serialization) → 3b (inference).** One design doc, two implementation plans. |
| Post-processing placement | **`alignair/postprocessing/` (numpy PP) + `alignair/inference/` (orchestration/decode).** |
| Port style | **Faithful port** — re-home the framework-agnostic numpy/pandas modules with lowercase names, light cleanup, and unit tests around depended-on behavior. |

## 2. Key finding (scopes the work)

~80% of the legacy inference/post-processing is **framework-agnostic numpy/pandas**
and is re-homed, not rewritten. The only TensorFlow coupling is (a) model loading
(`SavedModelInferenceWrapper`) and (b) the `.predict()` call. The new work is
therefore: a state_dict bundle, a thin PyTorch `Predictor` that reproduces the
**exact legacy output dict**, and faithful re-homes of the numpy PP modules.

## 3. Plan 3a — Serialization (`alignair/serialization/`)

- `bundle.py`
  - `ModelBundleConfig` / `TrainingMeta` dataclasses (reuse the legacy
    framework-agnostic schema: structural fields + provenance).
  - `save_bundle(dir, model_config, dataconfig, training_meta)` and
    `load_bundle(dir)`; `compute_fingerprint(dir)` (SHA-256 over bundle files).
  - On-disk bundle (replaces the TF SavedModel directory entirely):
    ```
    bundle_dir/
      model.pt            # torch state_dict
      model_config.json   # ModelConfig.to_dict()
      dataconfig.pkl      # optional: GenAIRR DataConfig (needed for AIRR/germline)
      training_meta.json  # TrainingMeta
      VERSION             # bundle format version
      fingerprint.txt     # SHA-256 of the above (excluding itself)
    ```
- `pretrained.py` — `PretrainedMixin` added to `BaseAlignAIR` (Single/Multi inherit):
  - `save_pretrained(bundle_dir, *, dataconfig=None, training_meta=None)` — writes
    the bundle from `self.config` (a `ModelConfig`) and `self.state_dict()`.
  - `from_pretrained(bundle_dir)` (classmethod) — reads `model_config.json`,
    reconstructs the correct class (`single_chain`/`multi_chain` via
    `ModelBundleConfig.model_type`) from `ModelConfig`, runs the lazy-param
    materialization forward, then `load_state_dict(model.pt)`. Returns the model.
  - `load_dataconfig(bundle_dir)` helper — returns the pickled `DataConfig` (or
    `None`) for the inference layer.

**Done when:** a trained model saves, reloads, and produces **identical forward
outputs** (state_dict round-trip parity); fingerprint is stable across save/load;
the multi-chain `chain_type` head round-trips too.

## 4. Plan 3b — Inference (`alignair/inference/` + `alignair/postprocessing/`)

### `alignair/inference/`
- `predictor.py` — `Predictor(model, device=None)`: eval + `torch.no_grad`;
  `predict(x: dict, batch_size=...) -> dict[str, np.ndarray]` reproducing the
  legacy output keys (`v/j/d_allele`, `v/j/d_start`, `v/j/d_end`, `*_start_logits`,
  `*_end_logits`, `mutation_rate`, `indel_count`, `productive`, and `chain_type`
  for multi-chain). Detaches to CPU numpy so downstream PP is framework-free.
- `decode.py` — faithful re-home of:
  - `clean_and_extract`: merge batch dicts, take `argmax` of boundary logits for
    integer positions (fallback to the expectation scalars when logits absent).
  - `segment_correction`: subtract the per-sequence center-pad offset, clamp to
    `[0, len(seq)]`, enforce `v_start ≤ v_end ≤ d_start ≤ d_end ≤ j_start ≤ j_end`.
- `pipeline.py` — orchestrates: sequences (via the Phase-2 reader / a FASTA reader)
  → tokenize (`CenterPaddedTokenizer`) → `Predictor.predict` → `decode` →
  allele thresholding → germline alignment → AIRR build → table. Returns a
  `pandas.DataFrame` (and can write CSV/AIRR TSV).

### `alignair/postprocessing/`
- `allele_selector.py` — `MaxLikelihoodPercentageThreshold.get_alleles(probs,
  percentage=0.21, cap=3)` → per-sequence selected allele names + likelihoods.
- `germline.py` — `HeuristicReferenceMatcher.match(sequences, starts, ends,
  alleles, indel_counts)` → per-sequence `{start_in_seq, end_in_seq, start_in_ref,
  end_in_ref}` (affine-cost heuristic alignment to germline references).
- `airr.py` — region/junction/quality enrichment (`locus`, `v/d/j_sequence`,
  `np1/np2_length`, `junction`, `junction_aa`, `junction_length`, `v_identity`,
  `stop_codon`, `vj_in_frame`).
- `serialize.py` — assemble the basic CSV columns and the full AIRR table from the
  decoded predictions + selections + germline alignments.

**Done when:** `from_pretrained` a bundle (with a `DataConfig`), run the pipeline on
`tests/data/test/sample_igh.csv`, and get a `DataFrame` with allele calls,
corrected coordinates, and AIRR enrichment columns for every input sequence.

## 5. Interfaces / data flow

```
sequences ──tokenize──▶ Predictor.predict ──▶ raw dict (np)
   ──decode──▶ positions+probs ──allele_selector──▶ calls+likelihoods
   ──segment_correction──▶ corrected coords
   ──germline.match──▶ ref alignments
   ──airr enrich + serialize──▶ AIRR DataFrame
```

The `Predictor` output dict is the single contract the numpy PP consumes; it is
byte-for-byte the legacy `SavedModelInferenceWrapper.predict` schema, so each
re-homed PP module is verifiable against legacy behavior on identical arrays.

## 6. Constraints, risks, notes

- **DataConfig required for AIRR/germline.** Germline alignment and AIRR
  enrichment need reference sequences from a GenAIRR `DataConfig`. Synthetic-trained
  bundles include one; CSV-trained bundles may not — those get calls + corrected
  coordinates but not germline/AIRR unless a `DataConfig` is supplied at load. Tests
  use `HUMAN_IGH_OGRDB`.
- **No weight parity.** Inference is validated structurally (shapes, valid calls,
  monotonic corrected coordinates, AIRR columns present) — not against specific
  trained-model predictions.
- **3b is large.** If, while writing the plan, the post-processing re-home proves
  too big for one plan, split 3b into 3b (predictor + decode + thresholding) and 3c
  (germline + AIRR + serialize). Decide at plan-writing time.
- **Orientation absent** in Phase 3 by decision; mis-oriented inputs predict poorly
  until orientation is added to the model architecture later.

## 7. Out of scope (later phases)

- Orientation detection/correction (folded into the model architecture later).
- CLI, HF hub, reporting/plots, and deleting `src/AlignAIR` (Phase 4).
- Genotype adjustment and IMGT name translation (optional legacy stages) unless
  trivially included; not required for the core AIRR output.
