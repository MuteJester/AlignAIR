# AlignAIR Phase 2b — GenAIRR Synthetic Dataset — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add an on-the-fly GenAIRR 2.2.0 synthetic dataset that produces the exact same `(x, y)` contract as the CSV dataset, so Phase-1 models can train on simulated data without files.

**Architecture:** A shared `build_xy` helper factors the per-sample tensor assembly out of `AlignAIRDataset`. A GenAIRR helper builds the allele vocabulary from a `DataConfig`. `experiment_presets` builds configured GenAIRR `Experiment`s (default: legacy "full augmentation"). `SyntheticDataset(IterableDataset)` streams `stream_records()`, normalizes each record, and reuses the tokenizer + `RecordAdapter` + `build_xy` path. Synchronous (single-process) generator — worker sharding and a multi-chain variant are deferred.

**Tech Stack:** Python 3.12, PyTorch 2.12 (CPU), GenAIRR 2.2.0 (local editable at `/home/thomas/Desktop/GenAIRR`; NOTE its `__version__` string wrongly reports `1.0.0` — verify by capability, not version string), pytest. Venv: `.venv/bin/python`. Tests run `PYTHONPATH=src .venv/bin/python -m pytest <path> -v`. Test dirs contain NO `__init__.py`.

---

## Verified GenAIRR 2.2.0 facts (probed, not assumed)

- Load a config: `import GenAIRR.data as gdata; cfg = gdata.HUMAN_IGH_OGRDB` → a `DataConfig`
  (`cfg.metadata.has_d` True; `number_of_v_alleles`=198, `number_of_d_alleles`=33, `number_of_j_alleles`=7;
  `cfg.allele_list('v')` → objects with `.name`).
- Fluent build (all keyword-only):
  `Experiment.on(cfg).recombine().mutate(model='s5f', rate=0.05).invert_d(prob=0.05)
  .end_loss_5prime(length=(0,25)).end_loss_3prime(length=(0,25)).polymerase_indels(count=(0,5))
  .sequencing_errors(rate=0.001).ambiguous_base_calls(count=(0,5)).compile()`
- `compiled.stream_records(n=None, seed=0)` → `Iterator[dict]` (n=None streams indefinitely). ~3 ms/record.
- Record dict fields used (per-sample): `sequence` (str, **lowercase** — uppercase before tokenizing),
  `v_call`/`d_call`/`j_call` (str, may be comma-separated for ambiguous GT),
  `v_sequence_start`/`v_sequence_end`/`d_*`/`j_*` (int), `mutation_rate` (float, naturally varied),
  `n_indels` (int, the indel count), `productive` (bool **or None** → coerce None to 0.0).
- D is always present in this recipe; empty `d_call` is mapped to `Short-D` defensively.

## The `(x, y)` contract (identical to Plan 2a)

Per-sample: `x={"tokenized_sequence": (L,) int64}`; `y` float32 arrays — `v/j(/d)_start`,`_end` `(1,)`;
`v/j(/d)_allele` `(n,)` multi-hot; `mutation_rate`,`indel_count`,`productive` `(1,)`.

## File structure (Plan 2b)

```
src/alignair/data/
  sample_builder.py     build_xy(tokens, rec, encoder, has_d)  [NEW, shared]
  dataset.py            refactor __getitem__ to use build_xy   [MODIFY]
  record_adapter.py     _indel_count: accept numeric            [MODIFY]
  genairr.py            assert_genairr_capable, allele_vocab_from_dataconfig  [NEW]
  experiment_presets.py full_augmentation / no_corruption / minimal          [NEW]
  synthetic.py          SyntheticDataset(IterableDataset)        [NEW]
tests/alignair/data/
  test_sample_builder.py test_genairr.py test_experiment_presets.py test_synthetic.py
tests/alignair/integration/
  test_train_synthetic.py
```

GenAIRR-dependent tests begin with `genairr = pytest.importorskip("GenAIRR")` so they skip if GenAIRR is unavailable.

---

## Task 1: `data/sample_builder.py` + numeric indel; refactor dataset

**Files:** Create `src/alignair/data/sample_builder.py`; Modify `src/alignair/data/record_adapter.py`,
`src/alignair/data/dataset.py`; Test `tests/alignair/data/test_sample_builder.py`

- [ ] **Step 1: Write the failing test**

`tests/alignair/data/test_sample_builder.py`:
```python
import numpy as np
from alignair.data.sample_builder import build_xy
from alignair.data.encoders import AlleleEncoder


def _encoder():
    enc = AlleleEncoder()
    enc.register_gene("V", ["V*01", "V*02"], sort=False)
    enc.register_gene("J", ["J*01"], sort=False)
    enc.register_gene("D", ["D*01", "Short-D"], sort=False)
    return enc


def _rec():
    return {
        "v_start": 1.0, "v_end": 10.0, "j_start": 20.0, "j_end": 30.0,
        "d_start": 12.0, "d_end": 15.0,
        "v_call_set": {"V*02"}, "j_call_set": {"J*01"}, "d_call_set": {"D*01"},
        "mutation_rate": 0.1, "indel_count": 2.0, "productive": 1.0,
    }


def test_build_xy_with_d():
    tokens = np.zeros(8, np.int64)
    x, y = build_xy(tokens, _rec(), _encoder(), has_d=True)
    assert x["tokenized_sequence"].shape == (8,)
    assert list(y["v_allele"]) == [0.0, 1.0]
    assert list(y["d_allele"]) == [1.0, 0.0]
    assert y["v_start"].tolist() == [1.0]
    assert y["indel_count"].tolist() == [2.0]
    assert set(y) >= {"v_start", "v_end", "j_start", "j_end", "d_start", "d_end",
                      "v_allele", "j_allele", "d_allele", "mutation_rate",
                      "indel_count", "productive"}


def test_build_xy_no_d_omits_d():
    enc = AlleleEncoder()
    enc.register_gene("V", ["V*01"], sort=False)
    enc.register_gene("J", ["J*01"], sort=False)
    rec = {"v_start": 0.0, "v_end": 5.0, "j_start": 6.0, "j_end": 9.0,
           "v_call_set": {"V*01"}, "j_call_set": {"J*01"},
           "mutation_rate": 0.0, "indel_count": 0.0, "productive": 0.0}
    x, y = build_xy(np.zeros(8, np.int64), rec, enc, has_d=False)
    assert "d_allele" not in y and "d_start" not in y
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement `sample_builder.py`**

```python
"""Shared per-sample (x, y) assembly from a canonical record + tokens."""
import numpy as np


def build_xy(tokens: np.ndarray, rec: dict, encoder, has_d: bool):
    x = {"tokenized_sequence": tokens}
    y = {
        "v_start": np.array([rec["v_start"]], np.float32),
        "v_end": np.array([rec["v_end"]], np.float32),
        "j_start": np.array([rec["j_start"]], np.float32),
        "j_end": np.array([rec["j_end"]], np.float32),
        "v_allele": encoder.encode("V", [rec["v_call_set"]])[0],
        "j_allele": encoder.encode("J", [rec["j_call_set"]])[0],
        "mutation_rate": np.array([rec["mutation_rate"]], np.float32),
        "indel_count": np.array([rec["indel_count"]], np.float32),
        "productive": np.array([rec["productive"]], np.float32),
    }
    if has_d:
        y["d_start"] = np.array([rec["d_start"]], np.float32)
        y["d_end"] = np.array([rec["d_end"]], np.float32)
        y["d_allele"] = encoder.encode("D", [rec["d_call_set"]])[0]
    return x, y
```

- [ ] **Step 4: Extend `_indel_count` in `record_adapter.py`** — add a numeric fast-path at the
top of the function body (before the dict/list branch):
```python
def _indel_count(item) -> float:
    if isinstance(item, bool):
        return float(int(item))
    if isinstance(item, (int, float)):
        return float(item)
    if isinstance(item, (dict, list, tuple)):
        return float(len(item))
    if isinstance(item, str) and item.strip():
        try:
            parsed = literal_eval(item)
            return float(len(parsed)) if isinstance(parsed, (dict, list, tuple)) else 0.0
        except Exception:
            return 0.0
    return 0.0
```
(The `bool` check precedes `int` because `bool` is a subclass of `int`.)

- [ ] **Step 5: Refactor `AlignAIRDataset.__getitem__`** in `dataset.py` to use `build_xy`.
Replace the body of `__getitem__` with:
```python
    def __getitem__(self, i: int):
        from .sample_builder import build_xy
        row = self.reader[i]
        tokens, pad_left = self.tokenizer.encode_and_pad(row["sequence"])
        rec = self.adapter.adapt(row, pad_left)
        return build_xy(tokens, rec, self.encoder, self.has_d)
```

- [ ] **Step 6: Run — expect PASS** (new + existing dataset/adapter tests still green).
```
PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/data -q
```
Expected: all PASS.

- [ ] **Step 7: Commit**
```bash
git add src/alignair/data/sample_builder.py src/alignair/data/record_adapter.py src/alignair/data/dataset.py tests/alignair/data/test_sample_builder.py
git commit -m "feat(alignair): extract build_xy, numeric indel counts (shared sample assembly)"
```

---

## Task 2: `data/genairr.py` — capability check + vocab from DataConfig

**Files:** Create `src/alignair/data/genairr.py`; Test `tests/alignair/data/test_genairr.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.data.genairr import assert_genairr_capable, allele_vocab_from_dataconfig


def test_capability_check_passes():
    assert_genairr_capable()  # must not raise on the installed 2.2.0 code


def test_vocab_from_dataconfig_human_igh():
    vocab = allele_vocab_from_dataconfig(gdata.HUMAN_IGH_OGRDB)
    assert len(vocab["V"]) == 198 and len(vocab["J"]) == 7
    assert vocab["D"][-1] == "Short-D"
    assert len(vocab["D"]) == 34  # 33 real + Short-D
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
"""GenAIRR 2.2.0 helpers. Verify capability, not the (mislabelled) version string."""


def assert_genairr_capable() -> None:
    """Ensure the GenAIRR fluent 2.x Experiment API is importable.

    The local GenAIRR build reports __version__ == '1.0.0' but is actually the
    2.2.0 codebase; we check for the capability (stream_records) instead.
    """
    from GenAIRR import Experiment
    if not hasattr(Experiment, "stream_records"):
        raise RuntimeError(
            "GenAIRR >= 2.2.0 required: Experiment.stream_records is missing")


def allele_vocab_from_dataconfig(dataconfig) -> dict:
    """Per-gene allele vocabulary from a GenAIRR DataConfig.

    D vocabulary is sorted unique names + 'Short-D' as the LAST entry (matching
    the encoder/loss convention that the last D column is the Short-D class).
    """
    vocab = {
        "V": sorted(a.name for a in dataconfig.allele_list("v")),
        "J": sorted(a.name for a in dataconfig.allele_list("j")),
    }
    if dataconfig.metadata.has_d:
        vocab["D"] = sorted(a.name for a in dataconfig.allele_list("d")) + ["Short-D"]
    return vocab
```

- [ ] **Step 4: Run — expect PASS** (2 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/data/genairr.py tests/alignair/data/test_genairr.py
git commit -m "feat(alignair): GenAIRR capability check + allele vocab from DataConfig"
```

---

## Task 3: `data/experiment_presets.py` — augmentation recipes

**Files:** Create `src/alignair/data/experiment_presets.py`; Test `tests/alignair/data/test_experiment_presets.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.data.experiment_presets import full_augmentation, no_corruption, minimal


def test_full_augmentation_streams_a_record():
    exp = full_augmentation(gdata.HUMAN_IGH_OGRDB)
    rec = next(exp.stream_records(n=1, seed=0))
    assert "sequence" in rec and "v_call" in rec
    assert "mutation_rate" in rec and "n_indels" in rec


def test_minimal_and_no_corruption_build():
    assert next(minimal(gdata.HUMAN_IGH_OGRDB).stream_records(n=1, seed=0))["sequence"]
    assert next(no_corruption(gdata.HUMAN_IGH_OGRDB).stream_records(n=1, seed=0))["sequence"]
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
"""GenAIRR experiment presets (compiled, ready to stream_records)."""
from .genairr import assert_genairr_capable


def _base(dataconfig):
    from GenAIRR import Experiment
    assert_genairr_capable()
    return Experiment.on(dataconfig).recombine()


def minimal(dataconfig):
    """Recombination + SHM only — no corruption. Returns a compiled experiment."""
    return _base(dataconfig).mutate(model="s5f", rate=0.05).compile()


def no_corruption(dataconfig):
    """Alias of minimal for clarity at call sites."""
    return minimal(dataconfig)


def full_augmentation(dataconfig, *, mutation_rate: float = 0.05, invert_d_prob: float = 0.05,
                      end_loss_5=(0, 25), end_loss_3=(0, 25), indel_count=(0, 5),
                      seq_error_rate: float = 0.001, ambiguous_count=(0, 5)):
    """Legacy-style full augmentation: SHM + 5'/3' loss + indels + sequencing
    errors + ambiguous bases (+ D-inversion if the chain has a D gene).

    Returns a compiled experiment whose ``stream_records`` yields AIRR dicts.
    """
    exp = _base(dataconfig).mutate(model="s5f", rate=mutation_rate)
    if dataconfig.metadata.has_d:
        exp = exp.invert_d(prob=invert_d_prob)
    exp = (exp.end_loss_5prime(length=end_loss_5)
              .end_loss_3prime(length=end_loss_3)
              .polymerase_indels(count=indel_count)
              .sequencing_errors(rate=seq_error_rate)
              .ambiguous_base_calls(count=ambiguous_count))
    return exp.compile()
```

- [ ] **Step 4: Run — expect PASS** (2 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/data/experiment_presets.py tests/alignair/data/test_experiment_presets.py
git commit -m "feat(alignair): GenAIRR experiment presets (full_augmentation/minimal)"
```

---

## Task 4: `data/synthetic.py` — SyntheticDataset

**Files:** Create `src/alignair/data/synthetic.py`; Test `tests/alignair/data/test_synthetic.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
import torch
from torch.utils.data import DataLoader
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.data.experiment_presets import full_augmentation
from alignair.data.genairr import allele_vocab_from_dataconfig
from alignair.data.synthetic import SyntheticDataset
from alignair.data.collate import align_collate

L = 576


def _dataset(n=16):
    cfg = gdata.HUMAN_IGH_OGRDB
    exp = full_augmentation(cfg)
    vocab = allele_vocab_from_dataconfig(cfg)
    return SyntheticDataset(exp, max_seq_length=L, has_d=True, allele_vocab=vocab, n=n, seed=0), vocab


def test_yields_contract_samples():
    ds, vocab = _dataset(n=4)
    x, y = next(iter(ds))
    assert x["tokenized_sequence"].shape == (L,)
    assert int(x["tokenized_sequence"].max()) <= 5 and int(x["tokenized_sequence"].min()) >= 0
    assert y["v_allele"].shape == (len(vocab["V"]),)
    assert y["d_allele"].shape == (len(vocab["D"]),)
    assert set(y) >= {"v_start", "v_end", "j_start", "j_end", "d_start", "d_end",
                      "v_allele", "j_allele", "d_allele", "mutation_rate",
                      "indel_count", "productive"}


def test_dataloader_batches():
    ds, vocab = _dataset(n=16)
    dl = DataLoader(ds, batch_size=4, collate_fn=align_collate)
    x, y = next(iter(dl))
    assert x["tokenized_sequence"].shape == (4, L)
    assert x["tokenized_sequence"].dtype == torch.long
    assert y["v_allele"].shape == (4, len(vocab["V"]))


def test_finite_n_count():
    ds, _ = _dataset(n=10)
    samples = list(iter(ds))
    assert len(samples) <= 10  # may drop over-length records (rare at L=576)
    assert len(samples) >= 1
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
"""On-the-fly GenAIRR synthetic dataset producing the AlignAIR (x, y) contract."""
import logging

from torch.utils.data import IterableDataset

from .tokenizer import CenterPaddedTokenizer
from .encoders import AlleleEncoder
from .record_adapter import RecordAdapter
from .sample_builder import build_xy

logger = logging.getLogger(__name__)


class SyntheticDataset(IterableDataset):
    """Stream GenAIRR records and adapt them to per-sample (x, y).

    Parameters
    ----------
    compiled_experiment : object with ``stream_records(n, seed)`` (from a preset).
    max_seq_length : int — sequences longer than this are dropped (logged).
    has_d : bool
    allele_vocab : dict {"V": [...], "J": [...], "D": [...]} (D ends with 'Short-D').
    n : int | None — records per epoch (None = infinite stream).
    seed : int — base seed; each __iter__ uses seed + epoch for fresh-but-reproducible data.

    Use with ``num_workers=0`` (single-process); worker sharding is not implemented.
    """

    def __init__(self, compiled_experiment, max_seq_length: int, has_d: bool,
                 allele_vocab: dict, n: int | None = None, seed: int = 0):
        self.experiment = compiled_experiment
        self.max_seq_length = max_seq_length
        self.has_d = has_d
        self.n = n
        self.seed = seed
        self._epoch = 0

        self.tokenizer = CenterPaddedTokenizer(max_length=max_seq_length)
        self.adapter = RecordAdapter(has_d=has_d)
        self.encoder = AlleleEncoder()
        for gene in (["V", "J"] + (["D"] if has_d else [])):
            self.encoder.register_gene(gene, allele_vocab[gene], sort=False)

    def __iter__(self):
        seed = self.seed + self._epoch
        self._epoch += 1
        dropped = 0
        for record in self.experiment.stream_records(n=self.n, seed=seed):
            seq = str(record["sequence"]).upper()
            if len(seq) > self.max_seq_length:
                dropped += 1
                continue
            tokens, pad_left = self.tokenizer.encode_and_pad(seq)
            row = dict(record)
            row["indels"] = record.get("n_indels", 0)
            if self.has_d and not row.get("d_call"):
                row["d_call"] = "Short-D"
            rec = self.adapter.adapt(row, pad_left)
            yield build_xy(tokens, rec, self.encoder, self.has_d)
        if dropped:
            logger.warning("SyntheticDataset dropped %d over-length sequences", dropped)
```

- [ ] **Step 4: Run — expect PASS** (3 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/data/synthetic.py tests/alignair/data/test_synthetic.py
git commit -m "feat(alignair): add GenAIRR SyntheticDataset (IterableDataset)"
```

---

## Task 5: Integration — train on synthetic stream; exports

**Files:** Create `tests/alignair/integration/test_train_synthetic.py`; Modify `src/alignair/data/__init__.py`

- [ ] **Step 1: Write the integration test**

```python
import pytest
import torch
from torch.utils.data import DataLoader
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.config.model_config import ModelConfig
from alignair.core.single_chain import SingleChainAlignAIR
from alignair.losses.hierarchical import AlignAIRLoss
from alignair.data.experiment_presets import full_augmentation
from alignair.data.genairr import allele_vocab_from_dataconfig
from alignair.data.synthetic import SyntheticDataset
from alignair.data.collate import align_collate
from alignair.training.config import TrainingConfig
from alignair.training.trainer import Trainer


def test_train_few_steps_on_synthetic():
    torch.manual_seed(0)
    cfg_dc = gdata.HUMAN_IGH_OGRDB
    vocab = allele_vocab_from_dataconfig(cfg_dc)
    ds = SyntheticDataset(full_augmentation(cfg_dc), max_seq_length=576, has_d=True,
                          allele_vocab=vocab, n=16, seed=0)
    loader = DataLoader(ds, batch_size=4, collate_fn=align_collate)

    cfg = ModelConfig(max_seq_length=576, v_allele_count=len(vocab["V"]),
                      j_allele_count=len(vocab["J"]), d_allele_count=len(vocab["D"]),
                      has_d_gene=True)
    trainer = Trainer(SingleChainAlignAIR(cfg), AlignAIRLoss(cfg),
                      TrainingConfig(lr=1e-4, steps_per_epoch=3))
    logs = trainer.train_epoch(loader)
    assert torch.isfinite(torch.tensor(logs["loss"]))
```

- [ ] **Step 2: Run — expect PASS.**
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/integration/test_train_synthetic.py -v`

- [ ] **Step 3: Add exports** — `src/alignair/data/__init__.py`:
```python
from .dataset import AlignAIRDataset, allele_vocab_from_csv
from .collate import align_collate
from .synthetic import SyntheticDataset
from .experiment_presets import full_augmentation, no_corruption, minimal
from .genairr import allele_vocab_from_dataconfig

__all__ = ["AlignAIRDataset", "allele_vocab_from_csv", "align_collate",
           "SyntheticDataset", "full_augmentation", "no_corruption", "minimal",
           "allele_vocab_from_dataconfig"]
```
NOTE: this makes `import alignair.data` import GenAIRR (via `synthetic`/`experiment_presets`/`genairr`).
That is acceptable in 2b (GenAIRR is a hard dependency of synthetic). If GenAIRR-free import of the CSV
path is later required, move the synthetic exports behind a lazy accessor.

- [ ] **Step 4: Run the full alignair suite — expect PASS.**
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair -q`

- [ ] **Step 5: Commit**
```bash
git add tests/alignair/integration/test_train_synthetic.py src/alignair/data/__init__.py
git commit -m "feat(alignair): train-on-synthetic integration test + data exports"
```

---

## Self-Review

**Spec coverage (Phase 2 design §5):** `experiment_presets` (full/minimal) → Task 3; `SyntheticDataset`
streaming via the shared adapter/build path → Tasks 1+4; GenAIRR 2.2.0 capability check (not version
string) → Task 2; comma-ambiguous calls preserved (RecordAdapter splits) → Task 4; train on synthetic →
Task 5. **Deferred (noted):** `MultiChainSyntheticDataset` (multi-chain per-chain producers + chain_type)
and a threaded producer/queue — both explicitly out of 2b scope; single-chain synchronous generator first.

**Placeholder scan:** none — every step has complete, probed code and real assertions.

**Type consistency:** `build_xy(tokens, rec, encoder, has_d)` signature identical across Tasks 1/4;
`allele_vocab_from_dataconfig`/`allele_vocab_from_csv` both return `{"V","J","D"}` dicts with `Short-D`
last; `full_augmentation` returns a compiled experiment with `stream_records` consumed identically in
Tasks 3/4/5; record fields (`sequence`,`n_indels`,`*_call`,`*_sequence_start/end`,`mutation_rate`,
`productive`) match the probed schema.

**Known notes:** `productive` may be `None` (RecordAdapter coerces to 0.0). Most synthetic samples are
non-productive (no `productive_only()` so the productivity head sees both classes). `mutation_rate` is
naturally varied by the s5f model. Use `num_workers=0`.
```
