# AlignAIR PyTorch Migration — Phase 2 (Data + Training) Design

**Date:** 2026-06-16
**Status:** Approved (structure + decisions); to be split into plans 2a then 2b
**Depends on:** Phase 1 (DL core) — complete on branch `pytorch-migration`
**Parent design:** `docs/superpowers/specs/2026-06-15-alignair-pytorch-migration-design.md`

## 1. Goal

Build the data and training subsystems of the new `src/alignair` package in
PyTorch: a re-homed, framework-agnostic data pipeline, a map-style CSV dataset, a
full-featured PyTorch trainer, and a GenAIRR-backed synthetic dataset. By the end
of Phase 2 the Phase-1 models can be trained both from CSV files and from
on-the-fly simulated data.

### Decisions (confirmed with user)

| Decision | Choice |
| --- | --- |
| Data pipeline code | **Re-home** the framework-agnostic tokenizer / encoders / readers / column schema into clean lowercase `alignair/data/` modules (not import-reuse from legacy). |
| Dataset style | **Map-style in-memory** `torch.utils.data.Dataset` for CSV; iterable streaming deferred. |
| Synthetic data | **Included in Phase 2** (sub-plan 2b), built on GenAIRR. |
| Trainer features | **Full-featured**: AMP, grad clipping, constraint application, checkpoint save/resume, early stopping, CSV/JSONL logging, progress bar. |
| Decomposition | **Two ordered sub-plans: 2a (data + trainer) then 2b (synthetic).** One design doc (this), two implementation plans. |
| Synthetic recipe | **Port the legacy "full augmentation" recipe** (SHM mutation + 5'/3' loss + indels + sequencing errors/N's + D-inversion) as a configurable default preset. |
| GenAIRR version | **2.2.0.** The local editable repo at `/home/thomas/Desktop/GenAIRR` (branch `clonal-lineage-simulation`, `v2.2.0-55-g592dbe8`, pyproject `2.2.0`) is the target. NOTE: its `GenAIRR.__version__` string wrongly reports `1.0.0` — the *code* is 2.2.0. Bump the AlignAIR dependency pin to `GenAIRR>=2.2.0`. |

## 2. The `(x, y)` batch contract (linchpin)

Every data source produces one identical contract, already matching the Phase-1
`BaseAlignAIR.forward` input and `AlignAIRLoss` `y_true` keys:

- `x = {"tokenized_sequence": LongTensor (B, L)}` — integer tokens in `[0, 5]`.
- `y` (all float32):
  - Boundaries `(B, 1)`: `v_start, v_end, j_start, j_end` (+`d_start, d_end` if D).
  - Alleles: `v_allele (B, n_v)`, `j_allele (B, n_j)` multi-hot (+`d_allele (B, n_d)` if D).
  - Analysis `(B, 1)`: `mutation_rate, indel_count, productive`.
  - Multi-chain: `chain_type (B, n_chains)` one-hot.

The trainer and model are agnostic to whether a batch came from CSV or GenAIRR.

## 3. `alignair/data/` — re-homed pipeline (Phase 2a + part of 2b)

All lowercase, concern-separated. Framework-agnostic numpy except `dataset.py`
and `synthetic.py` (torch).

- `tokenizer.py` — `CenterPaddedTokenizer`: vocab `{A:1, T:2, G:3, C:4, N:5, P:0}`
  (size 6, matching the model embedding). Center-pads to `max_length` (left =
  floor of pad, right = ceil); returns `(tokens, left_pad_offset)`. Unknown chars
  → N(5).
- `encoders.py` — `AlleleEncoder` (register sorted allele list per gene; multi-hot
  encode a set of calls; handles comma-separated ambiguous calls; "Short-D"
  synthetic class as last D column). `ChainTypeEncoder` (one-hot over chain types).
- `column_schema.py` — `ColumnSet(has_d)`: the required/optional CSV columns.
- `readers.py` — `CsvTableReader`: pandas read of CSV/TSV; column presence checks;
  **tolerates missing `productive` (default 1.0) and `indels` (default count 0)**
  since the sample CSVs lack them; logs what was defaulted.
- `record_adapter.py` — `RecordAdapter`: one raw row + tokenization pad offset →
  canonical per-sample dict: shifts gene start/end coords by the pad offset, parses
  indel count, splits comma-separated calls into sets, coerces productive to float.
- `dataset.py` — `AlignAIRDataset(torch.utils.data.Dataset)`: map-style. Loads a
  CSV via `CsvTableReader`, builds the `AlleleEncoder` from a `DataConfig` (or
  `MultiDataConfigContainer`), and `__getitem__(i)` returns one sample's
  `(x_np, y_np)` via tokenizer + adapter + encoders. Carries `has_d`/multi-chain
  flags from the config.
- `collate.py` — `align_collate(batch)`: stacks the per-sample dicts into batched
  tensors honoring the §2 contract; converts to torch tensors (tokens long, rest
  float32).

Re-home faithfully from the legacy modules; do not invent new behavior. Each gets
unit tests.

## 4. `alignair/training/` — full-featured trainer (Phase 2a)

- `config.py` — `TrainingConfig` dataclass: `epochs`, `lr`, `batch_size`,
  `weight_decay`, `use_amp`, `grad_clip_norm`, `steps_per_epoch` (optional),
  `checkpoint_dir`, `early_stopping_patience`, `log_every`, `seed`.
- `trainer.py` — `Trainer`: owns model, `AlignAIRLoss`, optimizer (Adam), optional
  AMP `GradScaler`. Training step: forward → loss → `scaler.scale(loss).backward()`
  → unscale → grad-clip → `scaler.step` → after `optimizer.step`, call
  `model.apply_constraints()` and each `loss.weights[*].apply_constraints()`.
  Per-epoch validation (no grad). Checkpoint **save** (model state_dict + optimizer
  + scaler + epoch + config + RNG state) and **resume**. Loss-component logging and
  optional Phase-1 metric accumulation (boundary acc, allele AUC, entropy).
  NaN/Inf guard (skip non-finite steps, warn).
- `callbacks.py` — composable callback protocol with hooks
  (`on_epoch_end`, `on_step_end`): `EarlyStopping`, `ModelCheckpoint` (best by
  monitored metric), `CSVLogger`, `JSONLinesLogger`, `ProgressBar` (tqdm).

Determinism helper (`seed_everything`) replacing the TF determinism module.

## 5. `alignair/data/synthetic.py` — GenAIRR synthetic dataset (Phase 2b)

- `experiment_presets.py` — builders returning a configured GenAIRR `Experiment`:
  `full_augmentation(dataconfig)` (default; SHM mutation model + 5'/3' end loss +
  polymerase indels + sequencing errors/ambiguous N bases + D-inversion),
  plus `no_corruption` and `minimal` presets. Built via the 2.2.0 fluent API:
  `Experiment.on(config).recombine().mutate(...).end_loss_5prime(...)
  .end_loss_3prime(...).polymerase_indels(...).sequencing_errors(...)
  .ambiguous_base_calls(...).compile()`.
- `synthetic.py` — `SyntheticDataset(torch.utils.data.IterableDataset)`: wraps a
  compiled experiment's `stream_records()`, adapts each GenAIRR record through the
  **same `RecordAdapter` + tokenizer + encoders + collate** path to the §2 contract.
  Comma-separated ambiguous calls are passed through as correct multi-allele GT
  (NOT reduced to first allele). Threaded producer with a bounded queue; timeout +
  worker-health guard to avoid deadlock.
- `MultiChainSyntheticDataset` — per-chain producers merged into one batch with the
  `chain_type` one-hot, mirroring `MultiChainDataset` partitioning.

## 6. Testing

- **2a unit:** tokenizer round-trip + center-pad offset; allele multi-hot incl.
  comma-separated + Short-D; chain-type one-hot; reader defaults for missing
  productive/indels; record-adapter coordinate shift; collate batch shapes/dtypes
  match the §2 contract.
- **2a integration:** construct `AlignAIRDataset` over `tests/data/test/sample_igh.csv`,
  wrap in a `DataLoader`, run the trainer for a handful of steps on a small model,
  assert total loss is finite and **decreases**; assert checkpoint save then resume
  restores state (loss continuity).
- **2b:** build `full_augmentation` over a small human IGH config, pull one batch
  from `SyntheticDataset`, validate the §2 contract (keys, shapes, dtypes, token
  range, multi-hot rows non-empty); run a few training steps without error.

## 7. Out of scope (later phases)

- Iterable streaming CSV for very large files (map-style only here).
- Serialization bundle format, inference pipeline, CLI, reporting (Phase 3+).
- New training scheme / curriculum beyond a standard supervised loop.

## 8. Risks & notes

- **Missing label columns:** sample CSVs lack `productive`/`indels`; the reader
  defaults them and logs it. Real training CSVs should provide them.
- **GenAIRR `__version__` mislabel:** code is 2.2.0 though the string says 1.0.0.
  Pin by capability, and add a smoke assertion that the 2.2.0 `Experiment` fluent
  API (e.g. `stream_records`) is present rather than trusting `__version__`.
- **GenAIRR API churn:** the local repo is 55 commits past the v2.2.0 tag on a
  feature branch; treat the importable API as source of truth and probe it in the
  2b plan before coding against assumed signatures.
