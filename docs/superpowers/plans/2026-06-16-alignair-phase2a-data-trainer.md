# AlignAIR Phase 2a — Data Pipeline + CSV Dataset + Trainer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the re-homed, framework-agnostic data pipeline (`alignair/data/`), a map-style CSV `AlignAIRDataset` producing the `(x, y)` contract, and a full-featured PyTorch `Trainer`, so a Phase-1 model can be trained from a CSV file with loss decreasing and checkpoint save/resume working.

**Architecture:** `alignair/data/` re-homes the legacy numpy tokenizer/encoders/readers/schema into clean lowercase modules and adds a torch `Dataset` + `collate`. `alignair/training/` adds a `TrainingConfig`, composable callbacks, and a `Trainer` (AMP, grad-clip, constraint application, checkpoint/resume, logging). Plan 2a is **GenAIRR-independent**: the dataset derives its allele vocabulary from an explicit vocab or by scanning the CSV — no `DataConfig` required.

**Tech Stack:** Python 3.12, PyTorch 2.12 (CPU), pandas, pytest. Venv: `.venv/bin/python`. Tests run from repo root: `PYTHONPATH=src .venv/bin/python -m pytest <path> -v`. Test dirs contain NO `__init__.py` (prepend import mode — an `alignair` package under `tests/` would shadow `src/alignair`).

---

## The `(x, y)` contract (target of the whole pipeline)

Per batch (`B` samples, sequence length `L`):

- `x = {"tokenized_sequence": LongTensor (B, L)}`, integer tokens in `[0, 5]`.
- `y` (float32 tensors):
  - `v_start, v_end, j_start, j_end` each `(B, 1)`; `+ d_start, d_end` if D.
  - `v_allele (B, n_v)`, `j_allele (B, n_j)` multi-hot; `+ d_allele (B, n_d)` if D.
  - `mutation_rate, indel_count, productive` each `(B, 1)`.
  - `chain_type (B, n_chains)` one-hot (multi-chain only — not exercised in 2a).

Per-sample (`__getitem__`) returns the same keys without the batch axis:
tokens `(L,)`; boundaries `(1,)`; alleles `(n,)`; scalars `(1,)`.

## Token vocabulary (must match the model embedding, vocab_size=6)

`{"A": 1, "T": 2, "G": 3, "C": 4, "N": 5, "P": 0}`; unknown char → N(5).

## File structure (Plan 2a)

```
src/alignair/data/
  __init__.py
  tokenizer.py        CenterPaddedTokenizer
  encoders.py         AlleleEncoder, ChainTypeEncoder
  column_schema.py    ColumnSet
  readers.py          CsvTableReader (tolerates missing productive/indels)
  record_adapter.py   RecordAdapter (per-sample canonical record)
  dataset.py          AlignAIRDataset (map-style) + allele_vocab_from_csv
  collate.py          align_collate
src/alignair/training/
  __init__.py
  config.py           TrainingConfig + seed_everything
  callbacks.py        Callback protocol + EarlyStopping/ModelCheckpoint/CSVLogger/ProgressBar
  trainer.py          Trainer
tests/alignair/data/  (no __init__.py)
  test_tokenizer.py test_encoders.py test_column_schema.py test_readers.py
  test_record_adapter.py test_dataset.py test_collate.py
tests/alignair/training/
  test_config.py test_callbacks.py test_trainer.py
tests/alignair/integration/
  test_train_csv.py
```

---

## Task 1: `data/tokenizer.py` — CenterPaddedTokenizer

**Files:** Create `src/alignair/data/tokenizer.py`; Test `tests/alignair/data/test_tokenizer.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from alignair.data.tokenizer import CenterPaddedTokenizer


def test_token_vocab():
    tok = CenterPaddedTokenizer(max_length=10)
    enc = tok.encode("ATGCN")
    assert list(enc) == [1, 2, 3, 4, 5]


def test_unknown_char_maps_to_n():
    tok = CenterPaddedTokenizer(max_length=10)
    assert list(tok.encode("AXZ")) == [1, 5, 5]


def test_center_pad_offsets():
    tok = CenterPaddedTokenizer(max_length=10)
    padded, pad_left = tok.encode_and_pad("ATGC")  # len 4, pad total 6 -> left 3 right 3
    assert padded.shape == (10,)
    assert pad_left == 3
    assert list(padded) == [0, 0, 0, 1, 2, 3, 4, 0, 0, 0]


def test_odd_padding_left_floor():
    tok = CenterPaddedTokenizer(max_length=8)
    padded, pad_left = tok.encode_and_pad("ATGCG")  # len 5, pad total 3 -> left 1 right 2
    assert pad_left == 1
    assert list(padded) == [0, 1, 2, 3, 4, 3, 0, 0]
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError`).
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/data/test_tokenizer.py -v`

- [ ] **Step 3: Implement**

```python
"""Center-padded nucleotide tokenizer (port of CenterPaddedSequenceTokenizer)."""
import numpy as np

TOKEN_DICT = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 5, "P": 0}


class CenterPaddedTokenizer:
    """Encode a nucleotide string to ints and center-pad to ``max_length``.

    Vocabulary size is 6 (0..5), matching the model embedding. Unknown chars map
    to N(5). Padding token is 0; left pad = floor(pad/2), right pad = ceil.
    """

    def __init__(self, max_length: int = 576, token_dict: dict | None = None):
        self.max_length = int(max_length)
        self.token_dict = token_dict or dict(TOKEN_DICT)
        self._n = self.token_dict["N"]

    def encode(self, sequence: str) -> np.ndarray:
        return np.array([self.token_dict.get(nt, self._n) for nt in sequence], dtype=np.int64)

    def encode_and_pad(self, sequence: str) -> tuple[np.ndarray, int]:
        encoded = self.encode(sequence)
        if len(encoded) > self.max_length:
            raise ValueError(
                f"sequence length {len(encoded)} exceeds max_length {self.max_length}")
        pad_total = self.max_length - len(encoded)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        padded = np.pad(encoded, (pad_left, pad_right), constant_values=0)
        return padded, pad_left
```

- [ ] **Step 4: Run — expect PASS** (4 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/data/tokenizer.py tests/alignair/data/test_tokenizer.py
git commit -m "feat(alignair): add CenterPaddedTokenizer (data pipeline re-home)"
```

---

## Task 2: `data/encoders.py` — AlleleEncoder + ChainTypeEncoder

**Files:** Create `src/alignair/data/encoders.py`; Test `tests/alignair/data/test_encoders.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from alignair.data.encoders import AlleleEncoder, ChainTypeEncoder


def test_multi_hot_single():
    enc = AlleleEncoder()
    enc.register_gene("V", ["A*01", "B*01", "C*01"], sort=False)
    ohe = enc.encode("V", [{"B*01"}])
    assert ohe.shape == (1, 3)
    assert list(ohe[0]) == [0.0, 1.0, 0.0]


def test_multi_hot_ambiguous_calls():
    enc = AlleleEncoder()
    enc.register_gene("V", ["A*01", "B*01", "C*01"], sort=False)
    ohe = enc.encode("V", [{"A*01", "C*01"}])
    assert list(ohe[0]) == [1.0, 0.0, 1.0]


def test_unknown_allele_ignored():
    enc = AlleleEncoder()
    enc.register_gene("V", ["A*01"], sort=False)
    ohe = enc.encode("V", [{"ZZZ*99"}])
    assert list(ohe[0]) == [0.0]


def test_chain_type_one_hot():
    enc = ChainTypeEncoder(["IGH", "IGK"])
    ohe = enc.encode(["IGK", "IGH"])
    assert ohe.shape == (2, 2)
    assert list(ohe[0]) == [0.0, 1.0]
    assert list(ohe[1]) == [1.0, 0.0]
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
"""Allele multi-hot and chain-type one-hot encoders (port of legacy encoders)."""
from dataclasses import dataclass

import numpy as np


@dataclass
class GeneEncoding:
    allele_to_index: dict
    index_to_allele: dict
    count: int


class AlleleEncoder:
    """Multi-hot encode sets of allele calls per gene type."""

    def __init__(self):
        self.gene_encodings: dict[str, GeneEncoding] = {}

    def register_gene(self, gene_type: str, allele_list, sort: bool = True,
                      allow_overwrite: bool = False) -> None:
        if gene_type in self.gene_encodings and not allow_overwrite:
            raise ValueError(f"Gene '{gene_type}' already registered")
        alleles = sorted(allele_list) if sort else list(allele_list)
        a2i = {a: i for i, a in enumerate(alleles)}
        i2a = {i: a for a, i in a2i.items()}
        self.gene_encodings[gene_type] = GeneEncoding(a2i, i2a, len(alleles))

    def count(self, gene_type: str) -> int:
        return self.gene_encodings[gene_type].count

    def encode(self, gene_type: str, allele_sets) -> np.ndarray:
        enc = self.gene_encodings[gene_type]
        rows = []
        for sample in allele_sets:
            ohe = np.zeros(enc.count, dtype=np.float32)
            for allele in sample:
                idx = enc.allele_to_index.get(allele)
                if idx is not None:
                    ohe[idx] = 1.0
            rows.append(ohe)
        return np.vstack(rows)


class ChainTypeEncoder:
    """One-hot encode chain-type labels in a fixed order."""

    def __init__(self, chain_types):
        self.chain_types = [str(c) for c in chain_types]
        self._index = {c: i for i, c in enumerate(self.chain_types)}

    @property
    def count(self) -> int:
        return len(self.chain_types)

    def encode(self, labels) -> np.ndarray:
        rows = []
        for label in labels:
            ohe = np.zeros(self.count, dtype=np.float32)
            idx = self._index.get(str(label))
            if idx is not None:
                ohe[idx] = 1.0
            rows.append(ohe)
        return np.vstack(rows)
```

- [ ] **Step 4: Run — expect PASS** (4 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/data/encoders.py tests/alignair/data/test_encoders.py
git commit -m "feat(alignair): add AlleleEncoder and ChainTypeEncoder"
```

---

## Task 3: `data/column_schema.py` — ColumnSet

**Files:** Create `src/alignair/data/column_schema.py`; Test `tests/alignair/data/test_column_schema.py`

- [ ] **Step 1: Write the failing test**

```python
from alignair.data.column_schema import ColumnSet


def test_no_d_columns():
    cs = ColumnSet(has_d=False)
    cols = cs.as_list()
    assert "sequence" in cols and "v_call" in cols and "j_call" in cols
    assert "d_call" not in cols
    assert "v_sequence_start" in cols and "j_sequence_end" in cols


def test_d_columns():
    cs = ColumnSet(has_d=True)
    cols = cs.as_list()
    assert "d_call" in cols and "d_sequence_start" in cols and "d_sequence_end" in cols


def test_label_vs_required():
    cs = ColumnSet(has_d=True)
    # productive/indels are optional labels (defaulted if absent)
    assert "productive" in cs.optional_columns
    assert "indels" in cs.optional_columns
    assert "sequence" in cs.required_columns
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
"""Required/optional column schema for AlignAIR CSV input."""
from dataclasses import dataclass


@dataclass
class ColumnSet:
    has_d: bool = True

    @property
    def required_columns(self) -> list[str]:
        cols = ["sequence", "v_call", "j_call",
                "v_sequence_start", "v_sequence_end",
                "j_sequence_start", "j_sequence_end", "mutation_rate"]
        if self.has_d:
            cols += ["d_call", "d_sequence_start", "d_sequence_end"]
        return cols

    @property
    def optional_columns(self) -> list[str]:
        # Defaulted when absent: productive -> 1.0, indels -> count 0.
        return ["productive", "indels"]

    def as_list(self) -> list[str]:
        return self.required_columns + self.optional_columns

    def __iter__(self):
        return iter(self.as_list())

    def __contains__(self, item) -> bool:
        return item in self.as_list()
```

- [ ] **Step 4: Run — expect PASS** (3 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/data/column_schema.py tests/alignair/data/test_column_schema.py
git commit -m "feat(alignair): add ColumnSet schema"
```

---

## Task 4: `data/readers.py` — CsvTableReader

**Files:** Create `src/alignair/data/readers.py`; Test `tests/alignair/data/test_readers.py`

Reads a CSV/TSV into a list of per-row dicts; validates required columns; defaults
missing optional columns (`productive` → 1.0, `indels` → "" so the adapter parses
count 0); logs what was defaulted.

- [ ] **Step 1: Write the failing test**

```python
import csv
import logging
import pytest
from alignair.data.readers import CsvTableReader
from alignair.data.column_schema import ColumnSet


def _write_csv(path, rows, header):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_reads_rows_and_len(tmp_path):
    p = tmp_path / "d.csv"
    header = ["sequence", "v_call", "j_call", "v_sequence_start", "v_sequence_end",
              "j_sequence_start", "j_sequence_end", "mutation_rate", "productive", "indels"]
    _write_csv(p, [{k: v for k, v in zip(header, ["ACGT", "V*01", "J*01", 0, 4, 5, 8, 0.1, "T", "{}"])}], header)
    reader = CsvTableReader(str(p), ColumnSet(has_d=False))
    assert len(reader) == 1
    row = reader[0]
    assert row["sequence"] == "ACGT"
    assert row["v_sequence_end"] == "4"  # raw string values; adapter coerces


def test_missing_required_raises(tmp_path):
    p = tmp_path / "d.csv"
    header = ["sequence", "v_call"]  # missing j_call etc.
    _write_csv(p, [{"sequence": "ACGT", "v_call": "V*01"}], header)
    with pytest.raises(ValueError):
        CsvTableReader(str(p), ColumnSet(has_d=False))


def test_defaults_missing_optional(tmp_path, caplog):
    p = tmp_path / "d.csv"
    header = ["sequence", "v_call", "j_call", "v_sequence_start", "v_sequence_end",
              "j_sequence_start", "j_sequence_end", "mutation_rate"]  # no productive/indels
    _write_csv(p, [{k: v for k, v in zip(header, ["ACGT", "V*01", "J*01", 0, 4, 5, 8, 0.1])}], header)
    with caplog.at_level(logging.WARNING):
        reader = CsvTableReader(str(p), ColumnSet(has_d=False))
    row = reader[0]
    assert row["productive"] == 1.0
    assert row["indels"] == ""
    assert any("default" in m.lower() for m in caplog.messages)
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
"""CSV/TSV reader producing per-row dicts; defaults missing optional columns."""
import logging
import pandas as pd

from .column_schema import ColumnSet

logger = logging.getLogger(__name__)

_DEFAULTS = {"productive": 1.0, "indels": ""}


class CsvTableReader:
    def __init__(self, path: str, column_set: ColumnSet, sep: str = ",",
                 nrows: int | None = None):
        df = pd.read_csv(path, sep=sep, nrows=nrows, dtype=str, keep_default_na=False)

        missing_required = [c for c in column_set.required_columns if c not in df.columns]
        if missing_required:
            raise ValueError(f"CSV missing required columns: {missing_required}")

        for col in column_set.optional_columns:
            if col not in df.columns:
                df[col] = _DEFAULTS[col]
                logger.warning("Column '%s' absent; defaulting to %r", col, _DEFAULTS[col])

        self.columns = column_set.as_list()
        self.records = df[self.columns].to_dict("records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        return self.records[i]
```

- [ ] **Step 4: Run — expect PASS** (3 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/data/readers.py tests/alignair/data/test_readers.py
git commit -m "feat(alignair): add CsvTableReader with optional-column defaults"
```

---

## Task 5: `data/record_adapter.py` — RecordAdapter

**Files:** Create `src/alignair/data/record_adapter.py`; Test `tests/alignair/data/test_record_adapter.py`

Converts a raw row + pad offset into a canonical per-sample dict: shifts gene
coordinates by `pad_left`, parses indel count, coerces productive, splits
comma-separated calls into sets.

- [ ] **Step 1: Write the failing test**

```python
from alignair.data.record_adapter import RecordAdapter


def _row(**over):
    base = {
        "sequence": "ACGT", "v_call": "V*01", "j_call": "J*01",
        "v_sequence_start": "0", "v_sequence_end": "4",
        "j_sequence_start": "5", "j_sequence_end": "8",
        "mutation_rate": "0.1", "productive": "T", "indels": "{}",
        "d_call": "D*01,D*02", "d_sequence_start": "4", "d_sequence_end": "5",
    }
    base.update(over)
    return base


def test_coord_shift_by_pad():
    ad = RecordAdapter(has_d=True)
    rec = ad.adapt(_row(), pad_left=3)
    assert rec["v_start"] == 3.0 and rec["v_end"] == 7.0
    assert rec["j_start"] == 8.0 and rec["j_end"] == 11.0
    assert rec["d_start"] == 7.0 and rec["d_end"] == 8.0


def test_ambiguous_calls_split_to_set():
    ad = RecordAdapter(has_d=True)
    rec = ad.adapt(_row(), pad_left=0)
    assert rec["d_call_set"] == {"D*01", "D*02"}
    assert rec["v_call_set"] == {"V*01"}


def test_indel_count_from_dict_string():
    ad = RecordAdapter(has_d=False)
    rec = ad.adapt(_row(indels="{'1': 5, '2': 9}"), pad_left=0)
    assert rec["indel_count"] == 2.0


def test_indel_count_empty():
    ad = RecordAdapter(has_d=False)
    assert ad.adapt(_row(indels=""), pad_left=0)["indel_count"] == 0.0


def test_productive_coercion():
    ad = RecordAdapter(has_d=False)
    assert ad.adapt(_row(productive="false"), pad_left=0)["productive"] == 0.0
    assert ad.adapt(_row(productive=1.0), pad_left=0)["productive"] == 1.0
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
"""Adapt a raw CSV row (+ pad offset) into a canonical per-sample record."""
from ast import literal_eval


def _to_float_bool(v) -> float:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return 1.0 if v != 0 else 0.0
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return 1.0
    if s in {"false", "0", "no", "n", "f", ""}:
        return 0.0
    return 0.0


def _indel_count(item) -> float:
    if isinstance(item, (dict, list, tuple)):
        return float(len(item))
    if isinstance(item, str) and item.strip():
        try:
            parsed = literal_eval(item)
            return float(len(parsed)) if isinstance(parsed, (dict, list, tuple)) else 0.0
        except Exception:
            return 0.0
    return 0.0


class RecordAdapter:
    def __init__(self, has_d: bool):
        self.has_d = has_d
        self.genes = ["v", "j"] + (["d"] if has_d else [])

    def adapt(self, row: dict, pad_left: int) -> dict:
        rec: dict = {}
        for g in self.genes:
            rec[f"{g}_start"] = float(int(row[f"{g}_sequence_start"]) + pad_left)
            rec[f"{g}_end"] = float(int(row[f"{g}_sequence_end"]) + pad_left)
            rec[f"{g}_call_set"] = set(str(row[f"{g}_call"]).split(","))
        rec["mutation_rate"] = float(row["mutation_rate"])
        rec["indel_count"] = _indel_count(row.get("indels", ""))
        rec["productive"] = _to_float_bool(row.get("productive", 1.0))
        return rec
```

- [ ] **Step 4: Run — expect PASS** (5 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/data/record_adapter.py tests/alignair/data/test_record_adapter.py
git commit -m "feat(alignair): add RecordAdapter (canonical per-sample record)"
```

---

## Task 6: `data/dataset.py` + `data/collate.py`

**Files:** Create `src/alignair/data/dataset.py`, `src/alignair/data/collate.py`;
Test `tests/alignair/data/test_dataset.py`, `tests/alignair/data/test_collate.py`

`AlignAIRDataset` is map-style: reads CSV via `CsvTableReader`, builds an
`AlleleEncoder` from an explicit `allele_vocab` or by scanning the CSV
(`allele_vocab_from_csv`). `__getitem__` returns per-sample `(x_np, y_np)`.
`align_collate` stacks samples into the batched `(x, y)` contract tensors.

- [ ] **Step 1: Write the failing test (collate)**

```python
import numpy as np
import torch
from alignair.data.collate import align_collate


def test_collate_stacks_contract():
    s1 = ({"tokenized_sequence": np.array([0, 1, 2, 0], np.int64)},
          {"v_start": np.array([1.0], np.float32),
           "v_allele": np.array([1.0, 0.0], np.float32),
           "mutation_rate": np.array([0.1], np.float32)})
    s2 = ({"tokenized_sequence": np.array([0, 3, 4, 0], np.int64)},
          {"v_start": np.array([2.0], np.float32),
           "v_allele": np.array([0.0, 1.0], np.float32),
           "mutation_rate": np.array([0.2], np.float32)})
    x, y = align_collate([s1, s2])
    assert x["tokenized_sequence"].shape == (2, 4)
    assert x["tokenized_sequence"].dtype == torch.long
    assert y["v_start"].shape == (2, 1)
    assert y["v_allele"].shape == (2, 2)
    assert y["v_start"].dtype == torch.float32
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement `collate.py`**

```python
"""Collate per-sample (x, y) dicts into batched tensors (the (x, y) contract)."""
import numpy as np
import torch


def align_collate(batch):
    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]

    x = {"tokenized_sequence": torch.as_tensor(
        np.stack([s["tokenized_sequence"] for s in xs]), dtype=torch.long)}

    y = {}
    for key in ys[0].keys():
        stacked = np.stack([s[key] for s in ys])
        y[key] = torch.as_tensor(stacked, dtype=torch.float32)
    return x, y
```

- [ ] **Step 4: Run collate test — expect PASS** (1 passed).

- [ ] **Step 5: Write the failing test (dataset)** — uses the repo sample CSV.

```python
import torch
from torch.utils.data import DataLoader
from alignair.data.dataset import AlignAIRDataset, allele_vocab_from_csv
from alignair.data.collate import align_collate

CSV = "tests/data/test/sample_igh.csv"


def test_vocab_from_csv_has_short_d_last():
    vocab = allele_vocab_from_csv(CSV, has_d=True)
    assert vocab["D"][-1] == "Short-D"
    assert len(vocab["V"]) == 198 and len(vocab["J"]) == 7
    assert len(vocab["D"]) == 35  # 34 unique + Short-D


def test_dataset_item_shapes():
    ds = AlignAIRDataset(CSV, max_seq_length=576, has_d=True, nrows=8)
    x, y = ds[0]
    assert x["tokenized_sequence"].shape == (576,)
    assert y["v_start"].shape == (1,)
    assert y["v_allele"].shape == (ds.v_allele_count,)
    assert y["d_allele"].shape == (ds.d_allele_count,)
    assert set(y.keys()) >= {"v_start", "v_end", "j_start", "j_end", "d_start",
                             "d_end", "v_allele", "j_allele", "d_allele",
                             "mutation_rate", "indel_count", "productive"}


def test_dataset_dataloader_batches():
    ds = AlignAIRDataset(CSV, max_seq_length=576, has_d=True, nrows=8)
    dl = DataLoader(ds, batch_size=4, collate_fn=align_collate)
    x, y = next(iter(dl))
    assert x["tokenized_sequence"].shape == (4, 576)
    assert y["v_allele"].shape == (4, ds.v_allele_count)
```

- [ ] **Step 6: Run — expect FAIL.**

- [ ] **Step 7: Implement `dataset.py`**

```python
"""Map-style CSV dataset producing the AlignAIR (x, y) contract."""
import numpy as np
from torch.utils.data import Dataset

from .column_schema import ColumnSet
from .readers import CsvTableReader
from .tokenizer import CenterPaddedTokenizer
from .encoders import AlleleEncoder
from .record_adapter import RecordAdapter


def allele_vocab_from_csv(csv_path: str, has_d: bool, sep: str = ",") -> dict:
    """Scan a CSV's call columns to build the per-gene allele vocabulary.

    D vocabulary is sorted unique calls + 'Short-D' as the LAST entry (the loss's
    short-D penalty reads the last D column).
    """
    reader = CsvTableReader(csv_path, ColumnSet(has_d=has_d), sep=sep)
    genes = {"V": "v_call", "J": "j_call"}
    if has_d:
        genes["D"] = "d_call"
    vocab: dict = {}
    for gene, col in genes.items():
        seen = set()
        for row in reader.records:
            for a in str(row[col]).split(","):
                if a:
                    seen.add(a)
        ordered = sorted(seen)
        if gene == "D":
            seen.discard("Short-D")
            ordered = sorted(seen) + ["Short-D"]
        vocab[gene] = ordered
    return vocab


class AlignAIRDataset(Dataset):
    def __init__(self, csv_path: str, max_seq_length: int, has_d: bool,
                 allele_vocab: dict | None = None, sep: str = ",",
                 nrows: int | None = None):
        self.has_d = has_d
        self.max_seq_length = max_seq_length
        self.reader = CsvTableReader(csv_path, ColumnSet(has_d=has_d), sep=sep, nrows=nrows)
        self.tokenizer = CenterPaddedTokenizer(max_length=max_seq_length)
        self.adapter = RecordAdapter(has_d=has_d)

        if allele_vocab is None:
            allele_vocab = allele_vocab_from_csv(csv_path, has_d=has_d, sep=sep)
        self.encoder = AlleleEncoder()
        for gene in (["V", "J"] + (["D"] if has_d else [])):
            self.encoder.register_gene(gene, allele_vocab[gene], sort=False)

        self.v_allele_count = self.encoder.count("V")
        self.j_allele_count = self.encoder.count("J")
        self.d_allele_count = self.encoder.count("D") if has_d else None

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, i: int):
        row = self.reader[i]
        tokens, pad_left = self.tokenizer.encode_and_pad(row["sequence"])
        rec = self.adapter.adapt(row, pad_left)

        x = {"tokenized_sequence": tokens}
        y = {
            "v_start": np.array([rec["v_start"]], np.float32),
            "v_end": np.array([rec["v_end"]], np.float32),
            "j_start": np.array([rec["j_start"]], np.float32),
            "j_end": np.array([rec["j_end"]], np.float32),
            "v_allele": self.encoder.encode("V", [rec["v_call_set"]])[0],
            "j_allele": self.encoder.encode("J", [rec["j_call_set"]])[0],
            "mutation_rate": np.array([rec["mutation_rate"]], np.float32),
            "indel_count": np.array([rec["indel_count"]], np.float32),
            "productive": np.array([rec["productive"]], np.float32),
        }
        if self.has_d:
            y["d_start"] = np.array([rec["d_start"]], np.float32)
            y["d_end"] = np.array([rec["d_end"]], np.float32)
            y["d_allele"] = self.encoder.encode("D", [rec["d_call_set"]])[0]
        return x, y
```

- [ ] **Step 8: Run dataset + collate tests — expect PASS** (4 passed total).
- [ ] **Step 9: Commit**
```bash
git add src/alignair/data/dataset.py src/alignair/data/collate.py tests/alignair/data/test_dataset.py tests/alignair/data/test_collate.py
git commit -m "feat(alignair): add map-style AlignAIRDataset and collate"
```

---

## Task 7: `training/config.py` — TrainingConfig + seed_everything

**Files:** Create `src/alignair/training/__init__.py` (empty), `src/alignair/training/config.py`;
Test `tests/alignair/training/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
import torch
from alignair.training.config import TrainingConfig, seed_everything


def test_defaults():
    cfg = TrainingConfig()
    assert cfg.epochs >= 1 and cfg.lr > 0
    assert cfg.use_amp in (True, False)


def test_roundtrip():
    cfg = TrainingConfig(epochs=3, lr=1e-3, batch_size=8)
    assert TrainingConfig.from_dict(cfg.to_dict()) == cfg


def test_seed_reproducible():
    seed_everything(123)
    a = torch.randn(4)
    seed_everything(123)
    b = torch.randn(4)
    assert torch.allclose(a, b)
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** (`config.py`; also create empty `training/__init__.py`)

```python
"""Training configuration and reproducibility helper."""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict

import numpy as np
import torch


@dataclass(eq=True)
class TrainingConfig:
    epochs: int = 1
    lr: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 0.0
    use_amp: bool = False
    grad_clip_norm: float = 10.0
    steps_per_epoch: int | None = None
    checkpoint_dir: str | None = None
    early_stopping_patience: int | None = None
    log_every: int = 10
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        return cls(**d)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

- [ ] **Step 4: Run — expect PASS** (3 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/training/__init__.py src/alignair/training/config.py tests/alignair/training/test_config.py
git commit -m "feat(alignair): add TrainingConfig and seed_everything"
```

---

## Task 8: `training/callbacks.py`

**Files:** Create `src/alignair/training/callbacks.py`; Test `tests/alignair/training/test_callbacks.py`

Composable callbacks with `on_epoch_end(epoch, logs)` and `on_train_end()`. Provide
`EarlyStopping`, `CSVLogger`, and a `CallbackList`. (ModelCheckpoint/ProgressBar are
exercised via the trainer in Task 9/10.)

- [ ] **Step 1: Write the failing test**

```python
import csv
from alignair.training.callbacks import EarlyStopping, CSVLogger, CallbackList


def test_early_stopping_triggers():
    es = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    es.on_epoch_end(0, {"val_loss": 1.0})
    assert not es.should_stop
    es.on_epoch_end(1, {"val_loss": 1.1})  # worse (1)
    es.on_epoch_end(2, {"val_loss": 1.2})  # worse (2) -> stop
    assert es.should_stop


def test_early_stopping_resets_on_improve():
    es = EarlyStopping(monitor="val_loss", patience=1, mode="min")
    es.on_epoch_end(0, {"val_loss": 1.0})
    es.on_epoch_end(1, {"val_loss": 1.1})  # worse (1)
    es.on_epoch_end(2, {"val_loss": 0.5})  # improve -> reset
    es.on_epoch_end(3, {"val_loss": 0.6})  # worse (1)
    assert not es.should_stop


def test_csv_logger_writes(tmp_path):
    p = tmp_path / "log.csv"
    logger = CSVLogger(str(p))
    logger.on_epoch_end(0, {"loss": 1.0, "val_loss": 0.9})
    logger.on_epoch_end(1, {"loss": 0.8, "val_loss": 0.7})
    logger.on_train_end()
    with open(p) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and rows[1]["loss"] == "0.8"


def test_callback_list_dispatches():
    es = EarlyStopping(monitor="val_loss", patience=0, mode="min")
    cl = CallbackList([es])
    cl.on_epoch_end(0, {"val_loss": 1.0})
    cl.on_epoch_end(1, {"val_loss": 1.1})
    assert cl.should_stop
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
"""Composable training callbacks."""
import csv


class Callback:
    def on_epoch_end(self, epoch: int, logs: dict) -> None: ...
    def on_train_end(self) -> None: ...


class EarlyStopping(Callback):
    def __init__(self, monitor: str = "val_loss", patience: int = 5, mode: str = "min"):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best = None
        self.wait = 0
        self.should_stop = False

    def _improved(self, current) -> bool:
        if self.best is None:
            return True
        return current < self.best if self.mode == "min" else current > self.best

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        if self.monitor not in logs:
            return
        current = logs[self.monitor]
        if self._improved(current):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.should_stop = True


class CSVLogger(Callback):
    def __init__(self, path: str):
        self.path = path
        self.rows: list[dict] = []

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        row = {"epoch": epoch}
        row.update({k: v for k, v in logs.items()})
        self.rows.append(row)
        self._flush()

    def _flush(self) -> None:
        fields = sorted({k for r in self.rows for k in r})
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)

    def on_train_end(self) -> None:
        self._flush()


class CallbackList(Callback):
    def __init__(self, callbacks):
        self.callbacks = list(callbacks)

    @property
    def should_stop(self) -> bool:
        return any(getattr(c, "should_stop", False) for c in self.callbacks)

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        for c in self.callbacks:
            c.on_epoch_end(epoch, logs)

    def on_train_end(self) -> None:
        for c in self.callbacks:
            c.on_train_end()
```

- [ ] **Step 4: Run — expect PASS** (4 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/training/callbacks.py tests/alignair/training/test_callbacks.py
git commit -m "feat(alignair): add training callbacks"
```

---

## Task 9: `training/trainer.py` — Trainer (+ AlignAIRLoss.apply_constraints)

**Files:** Create `src/alignair/training/trainer.py`; Modify
`src/alignair/losses/hierarchical.py` (add `apply_constraints`); Test
`tests/alignair/training/test_trainer.py`

The Trainer owns model + `AlignAIRLoss` + optimizer (+ optional AMP scaler). Step:
forward → loss → backward (scaled if AMP) → unscale → grad-clip → optimizer step →
`model.apply_constraints()` + `loss_fn.apply_constraints()`. Provides
`train_epoch`, `evaluate`, `fit` (with callbacks/early stop), `save_checkpoint`,
`load_checkpoint`.

- [ ] **Step 1: Add `apply_constraints` to AlignAIRLoss**

In `src/alignair/losses/hierarchical.py`, add this method to `AlignAIRLoss`:
```python
    @torch.no_grad()
    def apply_constraints(self) -> None:
        for w in self.weights.values():
            w.apply_constraints()
```

- [ ] **Step 2: Write the failing test**

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from alignair.config.model_config import ModelConfig
from alignair.core.single_chain import SingleChainAlignAIR
from alignair.losses.hierarchical import AlignAIRLoss
from alignair.training.config import TrainingConfig
from alignair.training.trainer import Trainer


def _tiny_model_and_loss():
    cfg = ModelConfig(max_seq_length=256, v_allele_count=4, j_allele_count=3,
                      d_allele_count=4, has_d_gene=True)
    return SingleChainAlignAIR(cfg), AlignAIRLoss(cfg), cfg


def _one_batch_loader(cfg, B=4):
    L = cfg.max_seq_length
    x = {"tokenized_sequence": torch.randint(0, 6, (B, L))}
    y = {
        "v_start": torch.full((B, 1), 1.0), "v_end": torch.full((B, 1), 100.0),
        "j_start": torch.full((B, 1), 120.0), "j_end": torch.full((B, 1), 200.0),
        "d_start": torch.full((B, 1), 105.0), "d_end": torch.full((B, 1), 110.0),
        "v_allele": torch.zeros(B, 4), "j_allele": torch.zeros(B, 3), "d_allele": torch.zeros(B, 4),
        "mutation_rate": torch.full((B, 1), 0.1), "indel_count": torch.full((B, 1), 1.0),
        "productive": torch.ones(B, 1),
    }
    y["v_allele"][:, 0] = 1.0; y["j_allele"][:, 0] = 1.0; y["d_allele"][:, 0] = 1.0
    # Single fixed batch repeated.
    return [(x, y)]


def test_single_train_step_returns_finite_loss():
    model, loss_fn, cfg = _tiny_model_and_loss()
    trainer = Trainer(model, loss_fn, TrainingConfig(lr=1e-3))
    (x, y), = _one_batch_loader(cfg)
    logs = trainer.train_step(x, y)
    assert torch.isfinite(torch.tensor(logs["loss"]))


def test_overfits_single_batch():
    torch.manual_seed(0)
    model, loss_fn, cfg = _tiny_model_and_loss()
    trainer = Trainer(model, loss_fn, TrainingConfig(lr=1e-3))
    (x, y), = _one_batch_loader(cfg)
    first = trainer.train_step(x, y)["loss"]
    for _ in range(15):
        last = trainer.train_step(x, y)["loss"]
    assert last < first  # loss decreases on a repeated batch


def test_checkpoint_save_and_resume(tmp_path):
    model, loss_fn, cfg = _tiny_model_and_loss()
    trainer = Trainer(model, loss_fn, TrainingConfig(lr=1e-3))
    (x, y), = _one_batch_loader(cfg)
    trainer.train_step(x, y)
    ckpt = tmp_path / "ck.pt"
    trainer.save_checkpoint(str(ckpt), epoch=1)

    model2, loss2, _ = _tiny_model_and_loss()
    trainer2 = Trainer(model2, loss2, TrainingConfig(lr=1e-3))
    state = trainer2.load_checkpoint(str(ckpt))
    assert state["epoch"] == 1
    # Weights match after load.
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
```

- [ ] **Step 3: Run — expect FAIL.**

- [ ] **Step 4: Implement `trainer.py`**

```python
"""PyTorch trainer for AlignAIR models."""
import logging

import torch

from .config import TrainingConfig, seed_everything
from .callbacks import CallbackList

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, loss_fn, config: TrainingConfig, device: str | None = None,
                 optimizer=None):
        self.config = config
        seed_everything(config.seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        params = list(self.model.parameters()) + list(self.loss_fn.parameters())
        self.optimizer = optimizer or torch.optim.Adam(
            params, lr=config.lr, weight_decay=config.weight_decay)
        self.use_amp = config.use_amp and self.device == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _to_device(self, d):
        return {k: v.to(self.device) for k, v in d.items()}

    def train_step(self, x, y) -> dict:
        self.model.train()
        x, y = self._to_device(x), self._to_device(y)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device, enabled=self.use_amp):
            y_pred = self.model(x["tokenized_sequence"]).as_dict()
            total, components = self.loss_fn(y, y_pred)
        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),
            self.config.grad_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.model.apply_constraints()
        self.loss_fn.apply_constraints()
        logs = {"loss": float(total.detach().cpu())}
        logs.update({k: float(v.cpu()) for k, v in components.items()})
        return logs

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()
        totals, n = {}, 0
        for x, y in loader:
            x, y = self._to_device(x), self._to_device(y)
            y_pred = self.model(x["tokenized_sequence"]).as_dict()
            total, components = self.loss_fn(y, y_pred)
            n += 1
            totals["val_loss"] = totals.get("val_loss", 0.0) + float(total.cpu())
        if n:
            totals = {k: v / n for k, v in totals.items()}
        return totals

    def train_epoch(self, loader) -> dict:
        running, steps = 0.0, 0
        for i, (x, y) in enumerate(loader):
            logs = self.train_step(x, y)
            running += logs["loss"]
            steps += 1
            if self.config.steps_per_epoch and steps >= self.config.steps_per_epoch:
                break
        return {"loss": running / max(steps, 1)}

    def fit(self, train_loader, val_loader=None, callbacks=None) -> list[dict]:
        cb = CallbackList(callbacks or [])
        history = []
        for epoch in range(self.config.epochs):
            train_logs = self.train_epoch(train_loader)
            logs = dict(train_logs)
            if val_loader is not None:
                logs.update(self.evaluate(val_loader))
            history.append(logs)
            cb.on_epoch_end(epoch, logs)
            if cb.should_stop:
                logger.info("Early stopping at epoch %d", epoch)
                break
        cb.on_train_end()
        return history

    def save_checkpoint(self, path: str, epoch: int) -> None:
        torch.save({
            "model": self.model.state_dict(),
            "loss_fn": self.loss_fn.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "config": self.config.to_dict(),
            "torch_rng": torch.get_rng_state(),
        }, path)

    def load_checkpoint(self, path: str) -> dict:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.loss_fn.load_state_dict(state["loss_fn"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"])
        return {"epoch": state["epoch"], "config": state["config"]}
```

- [ ] **Step 5: Run — expect PASS** (3 passed).
- [ ] **Step 6: Commit**
```bash
git add src/alignair/training/trainer.py src/alignair/losses/hierarchical.py tests/alignair/training/test_trainer.py
git commit -m "feat(alignair): add PyTorch Trainer (AMP, grad-clip, constraints, checkpoint/resume)"
```

---

## Task 10: Integration — train on the sample CSV; package exports

**Files:** Create `tests/alignair/integration/test_train_csv.py`; Modify
`src/alignair/data/__init__.py`, `src/alignair/training/__init__.py` (exports)

- [ ] **Step 1: Write the integration test**

```python
import torch
from torch.utils.data import DataLoader
from alignair.config.model_config import ModelConfig
from alignair.core.single_chain import SingleChainAlignAIR
from alignair.losses.hierarchical import AlignAIRLoss
from alignair.data.dataset import AlignAIRDataset
from alignair.data.collate import align_collate
from alignair.training.config import TrainingConfig
from alignair.training.trainer import Trainer

CSV = "tests/data/test/sample_igh.csv"


def test_train_on_sample_csv_loss_decreases(tmp_path):
    torch.manual_seed(0)
    ds = AlignAIRDataset(CSV, max_seq_length=576, has_d=True, nrows=16)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=align_collate)

    cfg = ModelConfig(max_seq_length=576, v_allele_count=ds.v_allele_count,
                      j_allele_count=ds.j_allele_count, d_allele_count=ds.d_allele_count,
                      has_d_gene=True)
    model = SingleChainAlignAIR(cfg)
    loss_fn = AlignAIRLoss(cfg)
    trainer = Trainer(model, loss_fn, TrainingConfig(lr=1e-3, epochs=1))

    # One batch, many steps -> loss should fall.
    x, y = next(iter(loader))
    first = trainer.train_step(x, y)["loss"]
    for _ in range(20):
        last = trainer.train_step(x, y)["loss"]
    assert last < first

    # Save + resume restores weights exactly.
    ckpt = tmp_path / "c.pt"
    trainer.save_checkpoint(str(ckpt), epoch=1)
    model2 = SingleChainAlignAIR(cfg)
    trainer2 = Trainer(model2, AlignAIRLoss(cfg), TrainingConfig(lr=1e-3))
    assert trainer2.load_checkpoint(str(ckpt))["epoch"] == 1
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_fit_runs_one_epoch():
    ds = AlignAIRDataset(CSV, max_seq_length=576, has_d=True, nrows=8)
    loader = DataLoader(ds, batch_size=4, collate_fn=align_collate)
    cfg = ModelConfig(max_seq_length=576, v_allele_count=ds.v_allele_count,
                      j_allele_count=ds.j_allele_count, d_allele_count=ds.d_allele_count,
                      has_d_gene=True)
    trainer = Trainer(SingleChainAlignAIR(cfg), AlignAIRLoss(cfg),
                      TrainingConfig(epochs=1, steps_per_epoch=2))
    history = trainer.fit(loader)
    assert len(history) == 1 and "loss" in history[0]
```

- [ ] **Step 2: Run — expect FAIL** (imports resolve, but run to confirm the
pipeline wires end-to-end; if a real failure appears, debug before continuing).
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/integration/test_train_csv.py -v`

- [ ] **Step 3: Add package exports**

`src/alignair/data/__init__.py`:
```python
from .dataset import AlignAIRDataset, allele_vocab_from_csv
from .collate import align_collate

__all__ = ["AlignAIRDataset", "allele_vocab_from_csv", "align_collate"]
```

`src/alignair/training/__init__.py`:
```python
from .config import TrainingConfig, seed_everything
from .trainer import Trainer

__all__ = ["TrainingConfig", "seed_everything", "Trainer"]
```

- [ ] **Step 4: Run the full alignair suite — expect PASS.**
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair -q`

- [ ] **Step 5: Commit**
```bash
git add src/alignair/data/__init__.py src/alignair/training/__init__.py tests/alignair/integration/test_train_csv.py
git commit -m "feat(alignair): integration train-on-CSV test + package exports"
```

---

## Self-Review

**Spec coverage:** §3 data modules → Tasks 1–6; §4 trainer/config/callbacks → Tasks 7–9;
§6 testing (unit + integration train-on-CSV + checkpoint resume) → Tasks 1–10. The
GenAIRR-independence claim is honored by `allele_vocab_from_csv` (no DataConfig).
Synthetic (§5) is Plan 2b, intentionally excluded.

**Placeholder scan:** none — every step has complete code and real assertions.

**Type consistency:** `(x, y)` keys identical across dataset/collate/trainer/loss;
`AlleleEncoder.count`, `register_gene` names consistent (Tasks 2/6); `Trainer`
methods (`train_step`/`evaluate`/`fit`/`save_checkpoint`/`load_checkpoint`) and
`AlignAIRLoss.apply_constraints` consistent across Tasks 9/10.

**Known notes:** AMP only activates on CUDA (CPU test env runs fp32 — `GradScaler(enabled=False)`).
`max_seq_length=576` exceeds the longest sample sequence (515); subsampling
(`nrows`) keeps the CPU integration test fast.
```
