# AlignAIR Phase 3b — Inference Core (predict → calls + coordinates) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Load a saved PyTorch model and predict on sequences to produce per-sequence allele calls and padding-corrected V/D/J coordinates — the core inference path (germline alignment + AIRR table are Plan 3c).

**Architecture:** `alignair/inference/predictor.py` wraps a model in eval/no-grad and returns the legacy output dict as numpy. `alignair/inference/decode.py` extracts integer boundary positions (argmax of logits) and applies center-pad correction + V≤D≤J ordering (faithful ports of the legacy CleanAndExtract + SegmentCorrection stages). `alignair/postprocessing/allele_selector.py` ports `MaxLikelihoodPercentageThreshold` (decoupled from DataConfig — uses an index→allele map). `alignair/inference/predict.py` orchestrates: sequences → tokenize → predict → decode → select → structured result.

**Tech Stack:** Python 3.12, PyTorch 2.12 (CPU), numpy, pytest. Venv: `.venv/bin/python`. Tests run `PYTHONPATH=src .venv/bin/python -m pytest <path> -v`. Test dirs contain NO `__init__.py`.

---

## File structure (Plan 3b)

```
src/alignair/postprocessing/
  __init__.py
  allele_selector.py   max_likelihood_threshold, select_alleles
src/alignair/inference/
  __init__.py
  predictor.py         Predictor (batched forward -> numpy dict)
  decode.py            extract_positions, correct_segments
  predict.py           predict_calls (orchestration) + PredictionResult
tests/alignair/postprocessing/  test_allele_selector.py
tests/alignair/inference/       test_decode.py test_predictor.py
tests/alignair/integration/     test_predict_calls.py
```

---

## Task 1: `postprocessing/allele_selector.py`

**Files:** Create `src/alignair/postprocessing/__init__.py` (empty), `src/alignair/postprocessing/allele_selector.py`;
Test `tests/alignair/postprocessing/test_allele_selector.py`

Port of `MaxLikelihoodPercentageThreshold.max_likelihood_percentage_threshold` + `get_alleles`,
decoupled from DataConfig: select indices whose likelihood ≥ `percentage * max`, sorted desc, capped.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from alignair.postprocessing.allele_selector import max_likelihood_threshold, select_alleles


def test_threshold_selects_above_fraction_of_max():
    probs = np.array([0.9, 0.5, 0.1, 0.05])
    idx, lik = max_likelihood_threshold(probs, percentage=0.21, cap=3)
    # threshold = 0.9*0.21 = 0.189 -> indices 0 (0.9) and 1 (0.5)
    assert list(idx) == [0, 1]
    assert np.allclose(lik, [0.9, 0.5])


def test_threshold_cap():
    probs = np.array([0.9, 0.85, 0.8, 0.75, 0.7])
    idx, _ = max_likelihood_threshold(probs, percentage=0.5, cap=3)
    assert len(idx) == 3  # capped
    assert list(idx) == [0, 1, 2]  # top 3 by likelihood


def test_select_alleles_maps_names():
    probs = np.array([[0.9, 0.1, 0.05], [0.1, 0.8, 0.7]])
    index_to_allele = {0: "A*01", 1: "B*01", 2: "C*01"}
    out = select_alleles(probs, index_to_allele, percentage=0.21, cap=3)
    assert out[0][0] == ["A*01"]
    assert set(out[1][0]) == {"B*01", "C*01"}
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

`src/alignair/postprocessing/__init__.py`: empty.

`src/alignair/postprocessing/allele_selector.py`:
```python
"""Allele selection by max-likelihood percentage threshold (port of
MaxLikelihoodPercentageThreshold), decoupled from DataConfig."""
import numpy as np


def max_likelihood_threshold(prediction: np.ndarray, percentage: float = 0.21, cap: int = 3):
    """Select indices with likelihood >= percentage*max, sorted desc, capped at `cap`."""
    max_index = int(np.argmax(prediction))
    threshold_value = prediction[max_index] * percentage
    indices = np.where(prediction >= threshold_value)[0]
    indices = indices[np.argsort(-prediction[indices])]
    if len(indices) > cap:
        indices = indices[:cap]
    return indices, prediction[indices]


def select_alleles(prob_matrix: np.ndarray, index_to_allele: dict,
                   percentage: float = 0.21, cap: int = 3):
    """For each row, return (allele_names, likelihoods)."""
    results = []
    for vec in prob_matrix:
        idx, lik = max_likelihood_threshold(vec, percentage=percentage, cap=cap)
        results.append(([index_to_allele[int(i)] for i in idx], lik))
    return results
```

- [ ] **Step 4: Run — expect PASS** (3 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/postprocessing/__init__.py src/alignair/postprocessing/allele_selector.py tests/alignair/postprocessing/test_allele_selector.py
git commit -m "feat(alignair): port allele selector (max-likelihood threshold)"
```

---

## Task 2: `inference/decode.py`

**Files:** Create `src/alignair/inference/__init__.py` (empty), `src/alignair/inference/decode.py`;
Test `tests/alignair/inference/test_decode.py`

Ports CleanAndExtract (argmax positions from logits) + SegmentCorrection (remove center-pad offset,
clamp, enforce V≤D≤J).

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from alignair.inference.decode import extract_positions, correct_segments


def test_extract_positions_argmax():
    L = 8
    pred = {}
    for g in ("v", "j", "d"):
        for b in ("start", "end"):
            logits = np.full((2, L), -10.0, np.float32)
            logits[:, 3] = 10.0  # argmax at 3
            pred[f"{g}_{b}_logits"] = logits
    pos = extract_positions(pred, has_d=True)
    assert pos["v_start"].tolist() == [3, 3]
    assert "d_end" in pos


def test_correct_segments_removes_padding_and_orders():
    # seq len 10, max_length 16 -> padding (16-10)//2 = 3
    sequences = ["A" * 10, "A" * 10]
    positions = {
        "v_start": np.array([3, 3]), "v_end": np.array([8, 8]),
        "d_start": np.array([7, 7]), "d_end": np.array([9, 9]),
        "j_start": np.array([10, 10]), "j_end": np.array([14, 14]),
    }
    corrected = correct_segments(positions, sequences, max_length=16, has_d=True)
    # padding removed: v_start 3-3=0, v_end 8-3=5
    assert corrected["v_start"].tolist() == [0, 0]
    assert corrected["v_end"].tolist() == [5, 5]
    # ordering: v_end <= d_start <= d_end <= j_start <= j_end, all within [0,10]
    assert (corrected["d_start"] >= corrected["v_end"]).all()
    assert (corrected["j_start"] >= corrected["d_end"]).all()
    assert (corrected["j_end"] <= 10).all()
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

`src/alignair/inference/__init__.py`: empty.

`src/alignair/inference/decode.py`:
```python
"""Decode raw model outputs into integer boundary positions, padding-corrected.

Faithful port of the legacy CleanAndExtractStage (argmax positions) and
SegmentCorrectionStage (remove center-pad offset, clamp, enforce V<=D<=J)."""
import numpy as np


def extract_positions(pred: dict, has_d: bool) -> dict:
    """Argmax of *_start_logits / *_end_logits -> integer positions (N,)."""
    genes = ["v", "j"] + (["d"] if has_d else [])
    out = {}
    for g in genes:
        out[f"{g}_start"] = np.argmax(pred[f"{g}_start_logits"], axis=-1).astype(np.int32)
        out[f"{g}_end"] = np.argmax(pred[f"{g}_end_logits"], axis=-1).astype(np.int32)
    return out


def correct_segments(positions: dict, sequences, max_length: int, has_d: bool) -> dict:
    """Remove per-sequence center padding, clamp to sequence bounds, enforce V<=D<=J."""
    paddings = np.array([(max_length - len(s)) // 2 for s in sequences], dtype=np.int32)
    seq_lengths = np.array([len(s) for s in sequences], dtype=np.int32)

    def sanitize(raw_start, raw_end):
        s = np.floor(np.squeeze(raw_start) - paddings).astype(np.int32)
        e = np.floor(np.squeeze(raw_end) - paddings).astype(np.int32)
        s = np.clip(s, 0, seq_lengths - 1)
        e = np.clip(e, 1, seq_lengths)
        e = np.maximum(e, s + 1)
        return s, e

    v_start, v_end = sanitize(positions["v_start"], positions["v_end"])
    j_start, j_end = sanitize(positions["j_start"], positions["j_end"])

    d_start = d_end = None
    if has_d and positions.get("d_start") is not None:
        d_start, d_end = sanitize(positions["d_start"], positions["d_end"])

    if d_start is not None:
        d_start = np.maximum(d_start, v_end)
        d_end = np.maximum(d_end, d_start + 1)
        j_start = np.maximum(j_start, d_end)
        j_end = np.maximum(j_end, j_start + 1)
    else:
        j_start = np.maximum(j_start, v_end)
        j_end = np.maximum(j_end, j_start + 1)

    corrected = {"v_start": v_start, "v_end": v_end, "j_start": j_start, "j_end": j_end}
    if d_start is not None:
        corrected["d_start"] = d_start
        corrected["d_end"] = d_end
    return corrected
```

- [ ] **Step 4: Run — expect PASS** (2 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/inference/__init__.py src/alignair/inference/decode.py tests/alignair/inference/test_decode.py
git commit -m "feat(alignair): port decode (argmax positions + segment correction)"
```

---

## Task 3: `inference/predictor.py`

**Files:** Create `src/alignair/inference/predictor.py`; Test `tests/alignair/inference/test_predictor.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import torch
from alignair.config.model_config import ModelConfig
from alignair.core.single_chain import SingleChainAlignAIR
from alignair.inference.predictor import Predictor


def _model():
    cfg = ModelConfig(max_seq_length=256, v_allele_count=5, j_allele_count=3,
                      d_allele_count=4, has_d_gene=True)
    return SingleChainAlignAIR(cfg), cfg


def test_predict_returns_numpy_dict_with_legacy_keys():
    model, cfg = _model()
    pred = Predictor(model)
    tokens = np.random.randint(0, 6, (6, cfg.max_seq_length))
    out = pred.predict(tokens, batch_size=4)
    for k in ("v_allele", "j_allele", "d_allele", "v_start_logits", "j_end_logits",
              "mutation_rate", "indel_count", "productive"):
        assert k in out
        assert isinstance(out[k], np.ndarray)
    assert out["v_allele"].shape == (6, cfg.v_allele_count)
    assert out["v_start_logits"].shape == (6, cfg.max_seq_length)


def test_predict_batches_match_single_pass():
    model, cfg = _model()
    pred = Predictor(model)
    tokens = np.random.randint(0, 6, (5, cfg.max_seq_length))
    a = pred.predict(tokens, batch_size=2)["v_allele"]
    b = pred.predict(tokens, batch_size=5)["v_allele"]
    assert np.allclose(a, b, atol=1e-6)
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
"""Predictor: batched eval/no-grad forward returning the legacy numpy output dict."""
import numpy as np
import torch


class Predictor:
    def __init__(self, model, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, tokenized, batch_size: int = 256) -> dict:
        tokens = torch.as_tensor(np.asarray(tokenized), dtype=torch.long)
        n = tokens.shape[0]
        chunks = []
        for i in range(0, n, batch_size):
            batch = tokens[i:i + batch_size].to(self.device)
            out = self.model(batch).as_dict()
            chunks.append({k: v.detach().cpu().numpy() for k, v in out.items()})

        keys = chunks[0].keys()
        return {k: np.concatenate([c[k] for c in chunks], axis=0) for k in keys}
```

- [ ] **Step 4: Run — expect PASS** (2 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/inference/predictor.py tests/alignair/inference/test_predictor.py
git commit -m "feat(alignair): add Predictor (batched numpy inference)"
```

---

## Task 4: `inference/predict.py` — orchestration + integration

**Files:** Create `src/alignair/inference/predict.py`; Modify `src/alignair/inference/__init__.py`;
Test `tests/alignair/integration/test_predict_calls.py`

Orchestrates tokenize → predict → decode → select into a `PredictionResult` per the input order.

- [ ] **Step 1: Write the integration test**

```python
import numpy as np
import pandas as pd
from alignair.config.model_config import ModelConfig
from alignair.core.single_chain import SingleChainAlignAIR
from alignair.data.dataset import allele_vocab_from_csv
from alignair.inference.predict import predict_calls

CSV = "tests/data/test/sample_igh.csv"


def test_predict_calls_on_sample_sequences(tmp_path):
    vocab = allele_vocab_from_csv(CSV, has_d=True)
    cfg = ModelConfig(max_seq_length=576, v_allele_count=len(vocab["V"]),
                      j_allele_count=len(vocab["J"]), d_allele_count=len(vocab["D"]),
                      has_d_gene=True)
    model = SingleChainAlignAIR(cfg)
    # save + reload through the bundle to exercise the real load path
    model.save_pretrained(tmp_path)
    model = SingleChainAlignAIR.from_pretrained(tmp_path)

    sequences = pd.read_csv(CSV, nrows=5)["sequence"].tolist()
    result = predict_calls(model, sequences, allele_vocab=vocab, max_seq_length=576)

    assert len(result.v_calls) == 5
    for i, seq in enumerate(sequences):
        assert len(result.v_calls[i]) >= 1            # at least one V call
        assert isinstance(result.v_calls[i][0], str)
        assert 0 <= result.v_start[i] <= result.v_end[i] <= len(seq)
        assert result.v_end[i] <= result.d_start[i] or result.d_start[i] >= result.v_end[i]
        assert result.j_end[i] <= len(seq)
        assert result.productive[i] in (True, False)


def test_predict_calls_no_d_omits_d():
    vocab = {"V": ["V*01", "V*02"], "J": ["J*01"]}
    cfg = ModelConfig(max_seq_length=256, v_allele_count=2, j_allele_count=1,
                      d_allele_count=None, has_d_gene=False)
    model = SingleChainAlignAIR(cfg)
    result = predict_calls(model, ["ACGT" * 10], allele_vocab=vocab, max_seq_length=256)
    assert result.d_calls is None
    assert len(result.v_calls) == 1
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

`src/alignair/inference/predict.py`:
```python
"""End-to-end call prediction: sequences -> allele calls + corrected coordinates."""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..data.tokenizer import CenterPaddedTokenizer
from ..data.encoders import AlleleEncoder
from ..postprocessing.allele_selector import select_alleles
from .predictor import Predictor
from .decode import extract_positions, correct_segments


@dataclass
class PredictionResult:
    v_calls: List[List[str]]
    j_calls: List[List[str]]
    d_calls: Optional[List[List[str]]]
    v_start: np.ndarray
    v_end: np.ndarray
    j_start: np.ndarray
    j_end: np.ndarray
    d_start: Optional[np.ndarray]
    d_end: Optional[np.ndarray]
    mutation_rate: np.ndarray
    indel_count: np.ndarray
    productive: np.ndarray


def predict_calls(model, sequences, *, allele_vocab: dict, max_seq_length: int,
                  percentage: float = 0.21, cap: int = 3, batch_size: int = 256) -> PredictionResult:
    has_d = model.config.has_d_gene
    tokenizer = CenterPaddedTokenizer(max_length=max_seq_length)
    tokens = np.stack([tokenizer.encode_and_pad(s.upper())[0] for s in sequences])

    pred = Predictor(model).predict(tokens, batch_size=batch_size)

    positions = extract_positions(pred, has_d)
    corrected = correct_segments(positions, sequences, max_seq_length, has_d)

    encoder = AlleleEncoder()
    for gene in (["V", "J"] + (["D"] if has_d else [])):
        encoder.register_gene(gene, allele_vocab[gene], sort=False)

    def names(gene_key, gene):
        i2a = encoder.gene_encodings[gene].index_to_allele
        return [calls for calls, _lik in select_alleles(pred[gene_key], i2a, percentage, cap)]

    productive = (np.squeeze(pred["productive"], -1) > 0.5)
    return PredictionResult(
        v_calls=names("v_allele", "V"),
        j_calls=names("j_allele", "J"),
        d_calls=names("d_allele", "D") if has_d else None,
        v_start=corrected["v_start"], v_end=corrected["v_end"],
        j_start=corrected["j_start"], j_end=corrected["j_end"],
        d_start=corrected.get("d_start"), d_end=corrected.get("d_end"),
        mutation_rate=np.squeeze(pred["mutation_rate"], -1),
        indel_count=np.squeeze(pred["indel_count"], -1),
        productive=productive,
    )
```

Modify `src/alignair/inference/__init__.py`:
```python
from .predictor import Predictor
from .predict import predict_calls, PredictionResult

__all__ = ["Predictor", "predict_calls", "PredictionResult"]
```

- [ ] **Step 4: Run — expect PASS** (2 passed).
- [ ] **Step 5: Run full suite.**
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair -q`
Expected: all green.

- [ ] **Step 6: Commit**
```bash
git add src/alignair/inference/predict.py src/alignair/inference/__init__.py tests/alignair/integration/test_predict_calls.py
git commit -m "feat(alignair): end-to-end predict_calls (calls + corrected coordinates)"
```

---

## Self-Review

**Spec coverage (design §4 core):** predictor (legacy output dict) → Task 3; decode (argmax + segment
correction) → Task 2; allele thresholding → Task 1; orchestration sequences→calls+coords → Task 4.
Germline alignment + AIRR builder + serialization are Plan 3c (explicitly deferred).

**Placeholder scan:** none — every step has complete code and real assertions.

**Type consistency:** `select_alleles(prob_matrix, index_to_allele, percentage, cap)` consistent across
Tasks 1/4; `extract_positions(pred, has_d)` / `correct_segments(positions, sequences, max_length, has_d)`
consistent across Tasks 2/4; `Predictor.predict(tokenized, batch_size) -> dict[str,np.ndarray]` consistent
across Tasks 3/4; `AlleleEncoder.gene_encodings[gene].index_to_allele` is the Phase-2 encoder API.

**Known notes:** model has untrained weights in tests, so calls/coords are structurally valid but not
biologically meaningful (no weight parity by design). `productive` squeezed to (N,) bool. The orchestration
takes an explicit `allele_vocab` (works for both CSV- and DataConfig-derived vocabularies).
```
