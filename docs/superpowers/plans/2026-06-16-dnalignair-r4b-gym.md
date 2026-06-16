# DNAlignAIR R4b — GenAIRR Training Gym — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the online GenAIRR "gym": a curriculum-driven `IterableDataset` that streams simulated sequences and emits, per sample, the full ground-truth bundle the unified model trains on (per-position region + state, germline + in-sequence coordinates, multi-label allele calls, orientation, aggregate counts) — with clear verbose curriculum reporting.

**Architecture:** Ground truth is derived directly from the GenAIRR `stream_records()` AIRR dict + the `ReferenceSet` germline sequences (no provenance-object wrangling). A `Curriculum` maps training progress → corruption parameters and a human-readable `describe()`. `AlignAIRGym(IterableDataset)` builds a GenAIRR `Experiment` at the current difficulty, streams records, and yields target bundles; `gym_collate` pads/stacks them (incl. multi-hot allele targets via the ReferenceSet). The initial gym is **forward-orientation only** (orientation augmentation + coordinate canonicalization is a later curriculum stage; the orientation target is identity for now).

**Tech Stack:** Python 3.12, PyTorch 2.12 (CPU), GenAIRR 2.2.0, pytest. Tests run `PYTHONPATH=src .venv/bin/python -m pytest <path> -v`. Test dirs have NO `__init__.py`. GenAIRR-dependent tests start with `pytest.importorskip("GenAIRR")`.

**Confirmed record fields (forward frame, V<D<J):** `sequence`, `v/d/j_call` (comma-sep multi-allele),
`v/d/j_sequence_start/end`, `v/d/j_germline_start/end`, `n_quality_errors`, `n_pcr_errors`, `n_indels`,
`mutation_rate`, `productive`, `rev_comp`.

**Per-position state (R4b cut):** `{germline, substitution}` derived by comparing the observed gene
segment to its germline allele over the aligned span (equal-length spans only; indel spans default to
germline and are covered by the aggregate `indel_count`). `insertion/deletion` per-position is a later
refinement; the 4-class state head simply gets no insertion/deletion examples yet.

---

## File structure (R4b)

```
src/alignair/gym/
  __init__.py
  targets.py      build_targets(record, reference_set, has_d) -> per-sample GT dict
  collate.py      gym_collate(batch, reference_set, has_d) -> batched tensors
  curriculum.py   Curriculum (progress -> corruption params + describe)
  gym.py          AlignAIRGym (IterableDataset, verbose) + build_experiment
tests/alignair/gym/test_targets.py
tests/alignair/gym/test_collate.py
tests/alignair/gym/test_curriculum.py
tests/alignair/integration/test_gym_stream.py
```

---

## Task 1: `gym/targets.py` — ground-truth extraction

**Files:** Create `src/alignair/gym/__init__.py` (empty), `src/alignair/gym/targets.py`;
Test `tests/alignair/gym/test_targets.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
import numpy as np
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from GenAIRR import Experiment
from alignair.reference.reference_set import ReferenceSet
from alignair.gym.targets import build_targets
from alignair.nn.region_head import REGION_INDEX
from alignair.nn.state_head import STATE_INDEX


def _record(seed=5):
    exp = (Experiment.on(gdata.HUMAN_IGH_OGRDB).recombine().mutate(model="s5f", rate=0.05)
           .end_loss_5prime(length=(0, 8)).compile())
    return next(exp.stream_records(n=1, seed=seed))


def test_targets_region_spans_match_coords():
    rec = _record()
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    t = build_targets(rec, rs, has_d=True)
    seq = rec["sequence"]
    assert t["tokens"].shape == (len(seq),)
    region = t["region_labels"]
    vs, ve = int(rec["v_sequence_start"]), int(rec["v_sequence_end"])
    # all positions inside the V span are labeled V
    assert (region[vs:ve] == REGION_INDEX["V"]).all()
    js, je = int(rec["j_sequence_start"]), int(rec["j_sequence_end"])
    assert (region[js:je] == REGION_INDEX["J"]).all()


def test_targets_coords_and_calls_and_scalars():
    rec = _record()
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    t = build_targets(rec, rs, has_d=True)
    assert t["germline"]["v"] == (int(rec["v_germline_start"]), int(rec["v_germline_end"]))
    assert t["inseq"]["v"] == (int(rec["v_sequence_start"]), int(rec["v_sequence_end"]))
    assert rec["v_call"].split(",")[0] in t["calls"]["V"]
    assert t["orientation_id"] == 0
    assert t["noise_count"] == rec["n_quality_errors"] + rec.get("n_pcr_errors", 0)
    assert abs(t["mutation_rate"] - rec["mutation_rate"]) < 1e-6
    assert t["indel_count"] == rec["n_indels"]
    assert t["productive"] in (0.0, 1.0)


def test_targets_state_marks_some_substitutions():
    rec = _record()
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    t = build_targets(rec, rs, has_d=True)
    state = t["state_labels"]
    # with ~5% SHM there should be at least one substitution in the V span (no-indel records)
    if rec["n_indels"] == 0:
        vs, ve = int(rec["v_sequence_start"]), int(rec["v_sequence_end"])
        assert (state[vs:ve] == STATE_INDEX["substitution"]).sum() >= 1
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

`src/alignair/gym/__init__.py`: empty.

`src/alignair/gym/targets.py`:
```python
"""Derive the per-sample ground-truth bundle from a GenAIRR stream_records dict
plus the ReferenceSet germline sequences (forward-orientation frame)."""
import numpy as np

from ..data.tokenizer import TOKEN_DICT
from ..nn.region_head import REGION_INDEX
from ..nn.state_head import STATE_INDEX

_GENES = ("v", "d", "j")


def _tok(seq: str) -> np.ndarray:
    n = TOKEN_DICT["N"]
    return np.array([TOKEN_DICT.get(c, n) for c in seq.upper()], dtype=np.int64)


def build_targets(record: dict, reference_set, has_d: bool) -> dict:
    seq = str(record["sequence"]).upper()
    L = len(seq)
    coords = {g: (int(record[f"{g}_sequence_start"]), int(record[f"{g}_sequence_end"]))
              for g in _GENES if f"{g}_sequence_start" in record}
    germ = {g: (int(record[f"{g}_germline_start"]), int(record[f"{g}_germline_end"]))
            for g in _GENES if f"{g}_germline_start" in record}

    vs, ve = coords["v"]
    js, je = coords["j"]

    # ---- region labels ----
    region = np.full(L, REGION_INDEX["pre"], dtype=np.int64)
    region[vs:ve] = REGION_INDEX["V"]
    if has_d and "d" in coords:
        ds, de = coords["d"]
        region[ve:ds] = REGION_INDEX["N1"]
        region[ds:de] = REGION_INDEX["D"]
        region[de:js] = REGION_INDEX["N2"]
    else:
        region[ve:js] = REGION_INDEX["N1"]
    region[js:je] = REGION_INDEX["J"]
    region[je:L] = REGION_INDEX["post"]

    # ---- per-position state (germline vs substitution over equal-length gene spans) ----
    state = np.zeros(L, dtype=np.int64)  # germline = 0
    for g in _GENES:
        if g not in coords:
            continue
        call = str(record[f"{g}_call"]).split(",")[0]
        ref = reference_set.gene(g.upper())
        idx = ref.index.get(call)
        if idx is None:
            continue
        ss, ee = coords[g]
        gs, ge = germ[g]
        obs = seq[ss:ee]
        gref = ref.sequences[idx][gs:ge]
        if len(obs) == len(gref):
            for k in range(len(obs)):
                if obs[k] != gref[k]:
                    state[ss + k] = STATE_INDEX["substitution"]

    calls = {g.upper(): set(str(record[f"{g}_call"]).split(","))
             for g in _GENES if f"{g}_call" in record}

    return {
        "tokens": _tok(seq),
        "region_labels": region,
        "state_labels": state,
        "germline": germ,
        "inseq": coords,
        "calls": calls,
        "orientation_id": 0,  # forward-only gym for now
        "noise_count": float(record["n_quality_errors"] + record.get("n_pcr_errors", 0)),
        "mutation_rate": float(record["mutation_rate"]),
        "indel_count": float(record["n_indels"]),
        "productive": 1.0 if record["productive"] else 0.0,
    }
```

- [ ] **Step 4: Run — expect PASS** (3 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/gym/__init__.py src/alignair/gym/targets.py tests/alignair/gym/test_targets.py
git commit -m "feat(alignair): gym target extraction from GenAIRR records + ReferenceSet"
```

---

## Task 2: `gym/collate.py` — batch the target bundles

**Files:** Create `src/alignair/gym/collate.py`; Test `tests/alignair/gym/test_collate.py`

Right-pads tokens / region / state to the batch max; builds the attention mask; multi-hot-encodes the
allele calls via the ReferenceSet; stacks coordinates and scalars. Padding label for region/state is
`-100` (ignored by `CrossEntropyLoss`).

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import torch
from alignair.gym.collate import gym_collate
from alignair.nn.region_head import REGION_INDEX


class _Gene:
    def __init__(self, names):
        self.names = names
        self.index = {n: i for i, n in enumerate(names)}


class _RS:
    def __init__(self):
        self.genes = {"V": _Gene(["v0", "v1", "v2"]), "J": _Gene(["j0", "j1"]),
                      "D": _Gene(["d0", "d1"])}
        self.has_d = True

    def gene(self, g):
        return self.genes[g.upper()]


def _sample(L, vcalls):
    return {
        "tokens": np.ones(L, np.int64),
        "region_labels": np.full(L, REGION_INDEX["V"], np.int64),
        "state_labels": np.zeros(L, np.int64),
        "germline": {"v": (0, 5), "d": (0, 3), "j": (0, 4)},
        "inseq": {"v": (0, L), "d": (1, 2), "j": (2, 3)},
        "calls": {"V": set(vcalls), "J": {"j0"}, "D": {"d0"}},
        "orientation_id": 0, "noise_count": 2.0, "mutation_rate": 0.1,
        "indel_count": 1.0, "productive": 1.0,
    }


def test_collate_pads_and_multihot():
    rs = _RS()
    batch = [_sample(6, ["v0"]), _sample(4, ["v1", "v2"])]
    out = gym_collate(batch, rs, has_d=True)
    assert out["tokens"].shape == (2, 6) and out["mask"].shape == (2, 6)
    assert out["mask"][1].tolist() == [True, True, True, True, False, False]
    # region padding label is -100
    assert (out["region_labels"][1, 4:] == -100).all()
    # multi-hot V calls
    assert out["v_allele"].shape == (2, 3)
    assert out["v_allele"][0].tolist() == [1.0, 0.0, 0.0]
    assert out["v_allele"][1].tolist() == [0.0, 1.0, 1.0]   # multi-label row
    assert out["orientation_id"].tolist() == [0, 0]
    assert out["noise_count"].shape == (2, 1)
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
"""Collate gym target bundles into batched tensors."""
import numpy as np
import torch

IGNORE = -100  # CrossEntropyLoss ignore_index for padded per-position labels


def _multihot(call_sets, gene_ref):
    n = len(gene_ref.names)
    out = torch.zeros(len(call_sets), n, dtype=torch.float32)
    for i, names in enumerate(call_sets):
        for nm in names:
            j = gene_ref.index.get(nm)
            if j is not None:
                out[i, j] = 1.0
    return out


def gym_collate(batch, reference_set, has_d: bool):
    B = len(batch)
    lmax = max(len(s["tokens"]) for s in batch)

    tokens = torch.zeros(B, lmax, dtype=torch.long)
    mask = torch.zeros(B, lmax, dtype=torch.bool)
    region = torch.full((B, lmax), IGNORE, dtype=torch.long)
    state = torch.full((B, lmax), IGNORE, dtype=torch.long)
    for i, s in enumerate(batch):
        n = len(s["tokens"])
        tokens[i, :n] = torch.from_numpy(s["tokens"])
        mask[i, :n] = True
        region[i, :n] = torch.from_numpy(s["region_labels"])
        state[i, :n] = torch.from_numpy(s["state_labels"])

    genes = ["v", "j"] + (["d"] if has_d else [])
    out = {"tokens": tokens, "mask": mask, "region_labels": region, "state_labels": state}
    for g in genes:
        gs = torch.tensor([s["germline"][g][0] for s in batch], dtype=torch.long)
        ge = torch.tensor([s["germline"][g][1] for s in batch], dtype=torch.long)
        ins = torch.tensor([s["inseq"][g][0] for s in batch], dtype=torch.long)
        ine = torch.tensor([s["inseq"][g][1] for s in batch], dtype=torch.long)
        out[f"{g}_germline_start"], out[f"{g}_germline_end"] = gs, ge
        out[f"{g}_start"], out[f"{g}_end"] = ins, ine
        out[f"{g}_allele"] = _multihot([s["calls"][g.upper()] for s in batch],
                                       reference_set.gene(g.upper()))

    out["orientation_id"] = torch.tensor([s["orientation_id"] for s in batch], dtype=torch.long)
    for key in ("noise_count", "mutation_rate", "indel_count", "productive"):
        out[key] = torch.tensor([[s[key]] for s in batch], dtype=torch.float32)
    return out
```

- [ ] **Step 4: Run — expect PASS** (1 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/gym/collate.py tests/alignair/gym/test_collate.py
git commit -m "feat(alignair): gym collate (pad + multi-hot calls + coords/scalars)"
```

---

## Task 3: `gym/curriculum.py` — difficulty schedule (verbose)

**Files:** Create `src/alignair/gym/curriculum.py`; Test `tests/alignair/gym/test_curriculum.py`

A `Curriculum` maps a progress fraction `p in [0,1]` to corruption parameters that ramp from easy to
hard, and exposes a human-readable `describe(p)` for verbose training output.

- [ ] **Step 1: Write the failing test**

```python
from alignair.gym.curriculum import Curriculum


def test_curriculum_ramps_and_describes():
    c = Curriculum()
    easy = c.params(0.0)
    hard = c.params(1.0)
    # harder at p=1: higher mutation rate cap, more trim, indels, seq errors
    assert hard["mutation_rate"] >= easy["mutation_rate"]
    assert hard["end_loss_5"][1] >= easy["end_loss_5"][1]
    assert hard["indel_count"][1] >= easy["indel_count"][1]
    assert hard["seq_error_rate"] >= easy["seq_error_rate"]
    d = c.describe(0.5)
    assert isinstance(d, str) and "stage" in d.lower()


def test_curriculum_stage_index():
    c = Curriculum(stages=5)
    assert c.stage(0.0) == 0
    assert c.stage(1.0) == 4
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
"""Curriculum: training progress -> GenAIRR corruption parameters (easy -> hard)."""


def _lerp(a, b, p):
    return a + (b - a) * p


class Curriculum:
    """Ramps corruption from clean (p=0) to fully corrupted (p=1)."""

    def __init__(self, stages: int = 5):
        self.stages = stages

    def params(self, p: float) -> dict:
        p = max(0.0, min(1.0, p))
        return {
            "mutation_rate": _lerp(0.005, 0.08, p),
            "end_loss_5": (0, int(round(_lerp(0, 25, p)))),
            "end_loss_3": (0, int(round(_lerp(0, 25, p)))),
            "indel_count": (0, int(round(_lerp(0, 5, p)))),
            "seq_error_rate": _lerp(0.0, 0.02, p),
            "ambiguous_count": (0, int(round(_lerp(0, 5, p)))),
        }

    def stage(self, p: float) -> int:
        p = max(0.0, min(1.0, p))
        return min(self.stages - 1, int(p * self.stages))

    def describe(self, p: float) -> str:
        pr = self.params(p)
        return (f"curriculum stage {self.stage(p) + 1}/{self.stages} (p={p:.2f}): "
                f"mut≤{pr['mutation_rate']:.3f}, trim≤{pr['end_loss_5'][1]}, "
                f"indel≤{pr['indel_count'][1]}, seq_err≤{pr['seq_error_rate']:.3f}, "
                f"N≤{pr['ambiguous_count'][1]}")
```

- [ ] **Step 4: Run — expect PASS** (2 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/gym/curriculum.py tests/alignair/gym/test_curriculum.py
git commit -m "feat(alignair): training curriculum (easy->hard) with verbose describe"
```

---

## Task 4: `gym/gym.py` — the IterableDataset (verbose) + integration

**Files:** Create `src/alignair/gym/gym.py`; Modify `src/alignair/gym/__init__.py`;
Test `tests/alignair/integration/test_gym_stream.py`

`AlignAIRGym` holds dataconfig(s), a `ReferenceSet`, and a `Curriculum`. `set_progress(p)` updates the
difficulty (and logs `curriculum.describe`). `__iter__` builds a GenAIRR experiment at the current params,
streams records, and yields target bundles. Verbose: logs the curriculum stage when difficulty changes
and every `log_every` samples.

- [ ] **Step 1: Write the integration test**

```python
import logging
import pytest
import torch
from torch.utils.data import DataLoader
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.gym.gym import AlignAIRGym
from alignair.gym.collate import gym_collate
from alignair.nn.region_head import REGIONS
from alignair.nn.state_head import STATES


def test_gym_streams_batches_with_full_targets(caplog):
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    gym = AlignAIRGym([gdata.HUMAN_IGH_OGRDB], rs, n=12, seed=0, log_every=4)
    with caplog.at_level(logging.INFO):
        gym.set_progress(0.5)   # mid curriculum -> should log the stage
    loader = DataLoader(gym, batch_size=4,
                        collate_fn=lambda b: gym_collate(b, rs, has_d=True))
    batch = next(iter(loader))
    B = batch["tokens"].shape[0]
    assert B == 4
    assert batch["mask"].dtype == torch.bool
    assert batch["region_labels"].shape == batch["tokens"].shape
    assert batch["state_labels"].shape == batch["tokens"].shape
    assert batch["v_allele"].shape == (B, len(rs.gene("V").names))
    assert batch["v_germline_start"].shape == (B,)
    assert batch["productive"].shape == (B, 1)
    # verbose curriculum line was logged
    assert any("curriculum stage" in m.lower() for m in caplog.messages)


def test_gym_difficulty_affects_generation():
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    gym = AlignAIRGym([gdata.HUMAN_IGH_OGRDB], rs, n=20, seed=1)
    gym.set_progress(0.0)
    clean = [t for t in gym]
    gym.set_progress(1.0)
    hard = [t for t in gym]
    # harder curriculum yields more sequencing noise on average
    avg_clean = sum(t["noise_count"] for t in clean) / len(clean)
    avg_hard = sum(t["noise_count"] for t in hard) / len(hard)
    assert avg_hard >= avg_clean
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
"""AlignAIRGym: online GenAIRR curriculum generator yielding GT target bundles."""
import logging

from torch.utils.data import IterableDataset

from .curriculum import Curriculum
from .targets import build_targets

logger = logging.getLogger(__name__)


def build_experiment(dataconfig, params):
    """Compile a GenAIRR experiment at the given curriculum params (forward orientation)."""
    from GenAIRR import Experiment
    exp = Experiment.on(dataconfig).recombine().mutate(model="s5f", rate=params["mutation_rate"])
    if dataconfig.metadata.has_d:
        exp = exp.invert_d(prob=0.05)
    exp = (exp.end_loss_5prime(length=params["end_loss_5"])
              .end_loss_3prime(length=params["end_loss_3"])
              .polymerase_indels(count=params["indel_count"])
              .sequencing_errors(rate=params["seq_error_rate"])
              .ambiguous_base_calls(count=params["ambiguous_count"]))
    return exp.compile()


class AlignAIRGym(IterableDataset):
    def __init__(self, dataconfigs, reference_set, n=None, seed=0,
                 curriculum=None, log_every=0):
        self.dataconfigs = list(dataconfigs)
        self.reference_set = reference_set
        self.n = n
        self.seed = seed
        self.curriculum = curriculum or Curriculum()
        self.log_every = log_every
        self._p = 0.0
        self._epoch = 0

    def set_progress(self, p: float) -> None:
        self._p = max(0.0, min(1.0, p))
        logger.info("Gym %s", self.curriculum.describe(self._p))

    def __iter__(self):
        params = self.curriculum.params(self._p)
        seed = self.seed + self._epoch
        self._epoch += 1
        # round-robin a dataconfig per epoch (single-config is the common case)
        dc = self.dataconfigs[self._epoch % len(self.dataconfigs)]
        has_d = dc.metadata.has_d
        exp = build_experiment(dc, params)
        count = 0
        for record in exp.stream_records(n=self.n, seed=seed):
            count += 1
            if self.log_every and count % self.log_every == 0:
                logger.info("Gym generated %d samples (%s)", count,
                            self.curriculum.describe(self._p))
            yield build_targets(record, self.reference_set, has_d=has_d)
```

- [ ] **Step 4: Run — expect PASS** (2 passed).

- [ ] **Step 5: Add exports** — `src/alignair/gym/__init__.py`:
```python
from .gym import AlignAIRGym, build_experiment
from .curriculum import Curriculum
from .collate import gym_collate
from .targets import build_targets

__all__ = ["AlignAIRGym", "build_experiment", "Curriculum", "gym_collate", "build_targets"]
```

- [ ] **Step 6: Run the full suite — expect PASS.**
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair -q`

- [ ] **Step 7: Commit**
```bash
git add src/alignair/gym/gym.py src/alignair/gym/__init__.py tests/alignair/integration/test_gym_stream.py
git commit -m "feat(alignair): AlignAIRGym curriculum IterableDataset (verbose) + exports"
```

---

## Self-Review

**Spec coverage (R4 design §4 gym):** GT extraction (region/state/coords/calls/orientation/counts) →
Task 1; batching incl. multi-hot calls → Task 2; curriculum (easy→hard) + verbose describe → Task 3;
the streaming IterableDataset with verbose logging → Task 4. The "done when a batch trains a step" is
exercised in R4c; here shapes + verbose are validated.

**Placeholder scan:** none — every step has complete code and real assertions.

**Type consistency:** `build_targets(record, reference_set, has_d) -> dict` keys consumed by
`gym_collate(batch, reference_set, has_d)` (Tasks 1/2/4); `Curriculum().params(p)` keys consumed by
`build_experiment(dataconfig, params)` (Tasks 3/4); `REGION_INDEX`/`STATE_INDEX` from the nn modules;
gym yields the same per-sample dict `gym_collate` expects.

**Known notes:** forward-orientation only for now (orientation target = 0; orientation augmentation +
coordinate canonicalization is a later curriculum stage). Per-position state is `{germline, substitution}`
over equal-length spans (indel spans default to germline; covered by aggregate `indel_count`). Padded
per-position labels use `-100` (CE ignore_index). Verbose: `set_progress` logs the curriculum stage and
the gym logs every `log_every` samples; R4c's trainer adds the live tqdm progress bar with loss/metrics
and the curriculum line.
```
