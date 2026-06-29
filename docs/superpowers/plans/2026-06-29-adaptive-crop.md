# Adaptive Short-Read Crop + Strata (Slice 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add **one-sided, germline-anchored** cropping (modeling adaptive/immunoSEQ multiplex-primer reads) + the adaptive IGH strata (FR1/FR2/FR3-anchored, J-anchored, reverse-strand × short), so the benchmark can evaluate the model on realistic short reads.

**Architecture:** A new `crop_one_sided(record, c0)` keeps the window `[c0, len]` (cut 5′, keep CDR3+J), recomputing all gene coords and **dropping genes fully cut off** — unlike the symmetric `crop_record` it does NOT force D/V-tail presence (the point of adaptive is the 5′-V is gone). Anchor helpers compute `c0` from a germline position (FR primer) or a 3′ length (J primer). `StratumSpec` gains an `anchor` knob; `generate.py` applies the one-sided crop when set. New adaptive strata use it.

**Tech Stack:** Python 3.12, GenAIRR, `pytest`. Venv: `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python`.

## Global Constraints

- Run via `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python` and `... -m pytest`. Never bare `python`.
- GenAIRR records expose `v/d/j_sequence_start/end` and `v/d/j_germline_start/end`; FR positions are NOT available — anchor on `v_germline_start` (the platform recomputes it on crop, `crop.py:57`).
- The one-sided path MUST NOT keep the symmetric crop's D-always-present / V-tail-FLANK invariants. A gene whose read span is fully outside `[c0, L]` is set to `None` (absent).
- Adaptive amplicons are a tight ~80–130 bp band, distinct from a RACE/full one-sided gradient — label strata accordingly (`tags`).
- This slice produces RAW short-read numbers; the coverage-conditioned observable-truth metric (so irreducible ambiguity isn't penalized) is the NEXT plan — note it in stratum descriptions, do not implement here.
- Git commit messages: **never** Co-Authored-By/Claude attribution. Do not modify `.claude/settings.json` or `docs/architecture/adoption_roadmap.md`.

---

## File Structure

- Modify `src/alignair/gym/crop.py` — add `crop_one_sided` + anchor-to-c0 helpers.
- Modify `src/alignair/benchmark/core/schema.py` — `StratumSpec.anchor` field.
- Modify `src/alignair/benchmark/generation/generate.py` — apply one-sided crop when `anchor` set.
- Modify `src/alignair/benchmark/generation/strata.py` — add adaptive strata.
- Test: `tests/alignair/gym/test_crop_one_sided.py`, `tests/alignair/benchmark/test_adaptive_strata.py`.

---

### Task 1: `crop_one_sided` + germline/J anchor helpers

**Files:**
- Modify: `src/alignair/gym/crop.py`
- Test: `tests/alignair/gym/test_crop_one_sided.py`

**Interfaces:**
- Produces: `anchor_c0(record, anchor) -> int` where `anchor` is `("v_germline", g_start:int)` or
  `("j", keep_len:int)`; and `crop_one_sided(record, c0:int) -> dict` keeping `[c0, len]`, recomputing
  coords, setting fully-cut genes to `None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/test_crop_one_sided.py
import pytest
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.gym.gym import build_experiment
from alignair.gym.crop import crop_one_sided, anchor_c0


def _clean_record(seed=1):
    exp = build_experiment(gdata.HUMAN_IGH_OGRDB, dict(
        mutation_rate=0.0, productive_only=False, end_loss_5=(0, 0), end_loss_3=(0, 0),
        indel_count=(0, 0), seq_error_rate=0.0, ambiguous_count=(0, 0)))
    return list(exp.stream_records(n=1, seed=seed))[0]


def test_v_germline_anchor_drops_5prime_v_keeps_cdr3_and_j():
    r = _clean_record()
    g_start = 220                                            # FR3-ish anchor in germline coords
    c0 = anchor_c0(r, ("v_germline", g_start))
    cr = crop_one_sided(r, c0)
    assert cr["v_sequence_start"] == 0                       # V now starts at read 0 (5' V gone)
    assert cr["v_germline_start"] >= g_start                 # anchored at/after the primer germline pos
    assert cr["j_sequence_start"] is not None                # J retained (3' side kept)
    # residual V is short (a tail), not the full V
    assert (cr["v_sequence_end"] - cr["v_sequence_start"]) < (r["v_sequence_end"] - r["v_sequence_start"])
    assert len(cr["sequence"]) == len(str(r["sequence"])) - c0


def test_j_anchored_keeps_3prime_len():
    r = _clean_record(2)
    c0 = anchor_c0(r, ("j", 100))                            # keep the 3'-most 100 bp
    cr = crop_one_sided(r, c0)
    assert len(cr["sequence"]) == min(100, len(str(r["sequence"])))
    assert cr["j_sequence_end"] == len(cr["sequence"])       # read ends at J end (J-anchored)


def test_fully_cut_gene_becomes_absent():
    r = _clean_record(3)
    c0 = int(r["j_sequence_start"])                          # cut everything before J -> V (and D) gone
    cr = crop_one_sided(r, c0)
    assert cr.get("v_sequence_start") is None                # V fully cut -> absent (no fabricated tail)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/gym/test_crop_one_sided.py -q`
Expected: FAIL (`ImportError: cannot import name 'crop_one_sided'`).

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/alignair/gym/crop.py
def anchor_c0(record: dict, anchor) -> int:
    """Read-coordinate crop start c0 for a one-sided (3'-keep) crop.
    ("v_germline", g_start): start where V reaches germline position g_start (an FR primer site).
    ("j", keep_len): keep the 3'-most keep_len bp (a J-anchored amplicon)."""
    seq = str(record["sequence"]); L = len(seq)
    kind, val = anchor
    if kind == "j":
        return max(0, L - int(val))
    if kind == "v_germline":
        vs = int(record["v_sequence_start"]); vgs = int(record["v_germline_start"])
        return min(L, max(0, vs + max(0, int(val) - vgs)))
    raise ValueError(f"unknown anchor {anchor!r}")


def crop_one_sided(record: dict, c0: int) -> dict:
    """Keep the window [c0, len] (cut the 5' end, retain CDR3 + J). Recompute all gene coords;
    a gene whose read span lies entirely before c0 is dropped (set to None) — NO has-D / V-tail
    invariant (an adaptive read legitimately loses its 5' V)."""
    seq = str(record["sequence"]); L = len(seq)
    c0 = max(0, min(int(c0), L))
    if c0 == 0:
        return record
    new = dict(record)
    new["sequence"] = seq[c0:L]
    for g in _GENES:
        if record.get(f"{g}_sequence_start") is None:
            continue
        ss, ee = int(record[f"{g}_sequence_start"]), int(record[f"{g}_sequence_end"])
        if ee <= c0:                                          # gene entirely 5' of the window -> absent
            for k in (f"{g}_sequence_start", f"{g}_sequence_end",
                      f"{g}_germline_start", f"{g}_germline_end", f"{g}_call"):
                new[k] = None
            continue
        gs, ge = int(record[f"{g}_germline_start"]), int(record[f"{g}_germline_end"])
        left = max(0, c0 - ss)                                # gene bases lost off the 5' end
        new[f"{g}_sequence_start"] = max(0, ss - c0)
        new[f"{g}_sequence_end"] = ee - c0
        new[f"{g}_germline_start"] = gs + left
        new[f"{g}_germline_end"] = ge
    return new
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/gym/test_crop_one_sided.py -q`
Expected: all PASS. (If a record's `v_germline_start` > 220 already, `anchor_c0` returns `vs`; pick a `g_start` below the record's V length — the test uses a clean full-V record so v_germline_start≈0.)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/gym/crop.py tests/alignair/gym/test_crop_one_sided.py
git commit -m "gym: one-sided germline/J-anchored crop (adaptive short reads; drops 5' V, no has-D invariant)"
```

---

### Task 2: `StratumSpec.anchor` + generate wiring + adaptive strata

**Files:**
- Modify: `src/alignair/benchmark/core/schema.py`
- Modify: `src/alignair/benchmark/generation/generate.py`
- Modify: `src/alignair/benchmark/generation/strata.py`
- Test: `tests/alignair/benchmark/test_adaptive_strata.py`

**Interfaces:**
- Consumes: `crop_one_sided`, `anchor_c0` (Task 1).
- Produces: `StratumSpec.anchor: tuple | None = None`; `generate.py` applies `crop_one_sided(record,
  anchor_c0(record, anchor))` when `anchor` is set (else the existing `crop_record(record, crop_to)`);
  `adaptive_igh_strata(n)` returning FR1/FR2/FR3-anchored + J-anchored(80–130 band) + reverse-strand strata.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/benchmark/test_adaptive_strata.py
import pytest
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.benchmark.core.schema import StratumSpec, BenchmarkSpec
from alignair.benchmark.generation.strata import adaptive_igh_strata
from alignair.benchmark.generation.generate import generate_benchmark
from alignair.reference.reference_set import ReferenceSet


def test_stratum_anchor_field_defaults_none():
    s = StratumSpec(name="x", n=1, progress=1.0)
    assert s.anchor is None


def test_adaptive_strata_generate_short_reads_with_dropped_5prime_v():
    strata = adaptive_igh_strata(n_per_scenario=6)
    names = {s.name for s in strata}
    assert {"adaptive_fr3", "adaptive_fr2", "adaptive_janchor"} <= names
    spec = BenchmarkSpec(name="adp", dataconfig_name="HUMAN_IGH_OGRDB", seed=1,
                         strata=tuple(s for s in strata if s.name == "adaptive_fr3"))
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    cases = generate_benchmark(spec, reference_set=rs)
    assert cases
    # FR3-anchored reads are short and start at V germline ~>=195 (5' V dropped)
    for c in cases:
        assert len(c.sequence) < 200
        v = c.genes.get("V")
        if v is not None and v.germline_start is not None:
            assert v.germline_start >= 150
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/benchmark/test_adaptive_strata.py -q`
Expected: FAIL (`StratumSpec` has no `anchor`; `adaptive_igh_strata` undefined).

- [ ] **Step 3: Implement**

In `src/alignair/benchmark/core/schema.py`, add to `StratumSpec` (after `crop_to`):

```python
    anchor: tuple | None = None     # one-sided crop: ("v_germline", g_start) or ("j", keep_len)
```

In `src/alignair/benchmark/generation/generate.py`, find where `crop_record(record, stratum.crop_to)`
is applied (lines ~192 and ~225) and make each crop site honor `anchor` first:

```python
        from ...gym.crop import crop_record, crop_one_sided, anchor_c0
        if stratum.anchor is not None:
            record = crop_one_sided(record, anchor_c0(record, stratum.anchor))
        elif stratum.crop_to is not None:
            record = crop_record(record, stratum.crop_to)
```

In `src/alignair/benchmark/generation/strata.py`, add:

```python
def adaptive_igh_strata(n_per_scenario: int = 200) -> tuple[StratumSpec, ...]:
    """Adaptive/immunoSEQ-style short reads: multiplex V-framework-primer (FR1/FR2/FR3) -> J-primer
    amplicons, modeled as one-sided germline-anchored crops (5' V truncated at the primer site).
    NOTE: raw allele accuracy here is a FLOOR until the coverage-conditioned observable-truth metric
    lands (short windows make many alleles observationally identical)."""
    base = dict(progress=0.8, param_overrides={"crop_prob": 0.0})
    return (
        StratumSpec(name="adaptive_fr1", n=n_per_scenario, anchor=("v_germline", 10),
                    description="FR1 multiplex-primer amplicon (most 5' V retained).",
                    tags=("adaptive", "fr1", "short"), **base),
        StratumSpec(name="adaptive_fr2", n=n_per_scenario, anchor=("v_germline", 80),
                    description="FR2 multiplex-primer amplicon.", tags=("adaptive", "fr2", "short"), **base),
        StratumSpec(name="adaptive_fr3", n=n_per_scenario, anchor=("v_germline", 200),
                    description="FR3 multiplex-primer amplicon (CDR3-proximal, little 5' V).",
                    tags=("adaptive", "fr3", "short"), **base),
        StratumSpec(name="adaptive_janchor", n=n_per_scenario, anchor=("j", 110),
                    description="J-anchored ~110bp amplicon (3'/J-primer protocols).",
                    tags=("adaptive", "j_anchored", "short"), **base),
        StratumSpec(name="adaptive_fr3_revcomp", n=n_per_scenario, anchor=("v_germline", 200),
                    orientation_ids=(1,),
                    description="Reverse-complement FR3 amplicon (orientation x short product cell).",
                    tags=("adaptive", "fr3", "short", "orientation"), **base),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/benchmark/test_adaptive_strata.py -q`
Expected: PASS. (If `generate_benchmark`'s signature differs, match it — inspect `generation/generate.py`'s `generate_benchmark` params; pass `reference_set=rs` as a keyword.)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/benchmark/core/schema.py src/alignair/benchmark/generation/generate.py \
        src/alignair/benchmark/generation/strata.py tests/alignair/benchmark/test_adaptive_strata.py
git commit -m "benchmark: adaptive IGH strata (FR1/FR2/FR3 + J-anchored + revcomp short) via one-sided crop"
```

---

## Done criteria

- `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/gym/test_crop_one_sided.py tests/alignair/benchmark/test_adaptive_strata.py -q` green.
- `adaptive_igh_strata` generates short, germline-anchored reads with the 5′ V truncated (and fully
  absent when cut past V); `StratumSpec.anchor` drives the one-sided crop in generation.
- A quick `bench_xattn.py` run including these strata produces per-stratum adaptive V numbers (a
  FLOOR pending the observable-truth metric).

## Follow-on plans (not this plan)
1. **Coverage-conditioned observable truth** (recompute the distinguishable allele set on the cropped
   window) + the called-given-segment-present conditional metric — so adaptive scores are fair.
2. Dynamic-genotype predictor protocol (per-stratum reference) for partial_genotype/novel_allele.
3. Chimera/junction-indel/adapter/short-contaminant strata; IGK/IGL/TRB scaffolds.
