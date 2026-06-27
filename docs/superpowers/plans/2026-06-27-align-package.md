# `align/` Package Implementation Plan (Speed Stack — Phase 1a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained classical-alignment package (`src/alignair/align/`) — a pluggable aligner backend (parasail now, WFA2 behind the same interface), a non-learned k-mer seed prefilter, and a batched executor — that the inference rewrite will use to replace the differentiable soft-DP.

**Architecture:** A backend protocol returns a uniform `AlignResult(score, cigar, q_start, q_end, t_start, t_end)` from aligning a read-segment (query, globally consumed) to a germline (target, ends-free). The parasail backend factors the proven logic already in `src/alignair/io/alignment.py`. A k-mer `SeedPrefilter` admits divergent candidate alleles into the alignment pool independent of the neural encoder. A threaded `align_batch` runs many (query, target) pairs preserving input order.

**Tech Stack:** Python 3.12, `parasail` (installed), `pywfa` (to be installed; optional, graceful fallback), `pytest`. Venv: `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python`.

## Global Constraints

- Run everything via `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python` and `... -m pytest`. Never bare `python`.
- This package is **pure classical/CPU** — no torch, no model, no GPU imports anywhere in `src/alignair/align/`.
- `AlignResult.score`: **higher is better** (every backend normalizes to this).
- `AlignResult.cigar`: **core M/I/D ops only** (run-length, e.g. `4M1I4M`). Soft-clips (`S`) and germline-skip (`N`) are the caller's job, NOT the backend's — the backend reports them via `t_start`/`q_start` instead.
- Alignment mode is fixed: **query global, germline (target) ends free** (parasail `sg_dx`, WFA `ends-free` on text). Match `+2`, mismatch `-1`, gap-open `3`, gap-extend `1` (the values already used in `io/alignment.py`).
- Git commit messages: **never** include Co-Authored-By or any Claude/AI attribution (project rule).
- Do not modify `.claude/settings.json` or `docs/architecture/adoption_roadmap.md`.

---

## File Structure

- Create `src/alignair/align/__init__.py` — exports `AlignResult`, `Aligner`, `get_aligner`, `SeedPrefilter`, `align_batch`.
- Create `src/alignair/align/backend.py` — `AlignResult` dataclass, `Aligner` protocol, `get_aligner()` selector.
- Create `src/alignair/align/parasail.py` — `ParasailAligner` (factors `io/alignment.py` logic).
- Create `src/alignair/align/seed_prefilter.py` — `SeedPrefilter` k-mer admission.
- Create `src/alignair/align/batch.py` — `align_batch` threaded executor.
- Create `src/alignair/align/wfa.py` — `WFAAligner` (pywfa, optional).
- Create tests under `tests/alignair/align/`.

---

### Task 1: Backend protocol + result type

**Files:**
- Create: `src/alignair/align/backend.py`
- Create: `src/alignair/align/__init__.py`
- Test: `tests/alignair/align/test_backend.py`

**Interfaces:**
- Produces: `AlignResult` (frozen dataclass: `score: float, cigar: str, q_start: int, q_end: int, t_start: int, t_end: int`); `Aligner` Protocol with `align(self, query: str, target: str) -> AlignResult | None`; `get_aligner(prefer: str = "wfa") -> Aligner`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/align/test_backend.py
import pytest
from alignair.align.backend import AlignResult, get_aligner


def test_align_result_fields():
    r = AlignResult(score=16.0, cigar="8M", q_start=0, q_end=8, t_start=0, t_end=8)
    assert r.score == 16.0 and r.cigar == "8M"
    assert r.q_start == 0 and r.q_end == 8 and r.t_start == 0 and r.t_end == 8


def test_get_aligner_returns_something_with_align():
    al = get_aligner()                       # falls back to parasail if wfa absent
    assert hasattr(al, "align")
    r = al.align("ACGTACGT", "ACGTACGT")
    assert r is not None and r.cigar == "8M"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/align/test_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'alignair.align'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/align/backend.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class AlignResult:
    score: float        # higher is better
    cigar: str          # core M/I/D run-length ops only (no S/N)
    q_start: int        # first query base consumed (0-based)
    q_end: int          # one past last query base
    t_start: int        # germline (target) bases consumed before the core
    t_end: int          # germline end (exclusive)


@runtime_checkable
class Aligner(Protocol):
    def align(self, query: str, target: str) -> AlignResult | None: ...


def get_aligner(prefer: str = "wfa") -> Aligner:
    """Best available aligner. Tries WFA (pywfa) first when prefer=='wfa', else parasail."""
    if prefer == "wfa":
        try:
            from .wfa import WFAAligner
            return WFAAligner()
        except Exception:
            pass
    from .parasail import ParasailAligner
    return ParasailAligner()
```

```python
# src/alignair/align/__init__.py
from .backend import AlignResult, Aligner, get_aligner
from .parasail import ParasailAligner
from .seed_prefilter import SeedPrefilter
from .batch import align_batch

__all__ = ["AlignResult", "Aligner", "get_aligner", "ParasailAligner",
           "SeedPrefilter", "align_batch"]
```

Note: `__init__.py` imports modules created in later tasks. Create empty placeholder files now so the import works, OR implement Task 1 last in your editor — but the recommended order is to create the three referenced modules as empty stubs (`# stub`) now and fill them in Tasks 2–4. Simplest: comment out the `parasail`/`seed_prefilter`/`batch` imports in `__init__.py` until those tasks land, uncommenting each as you go.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/align/test_backend.py::test_align_result_fields -v`
Expected: PASS. (`test_get_aligner_returns_something_with_align` passes after Task 2.)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/align/backend.py src/alignair/align/__init__.py tests/alignair/align/test_backend.py
git commit -m "align: backend protocol + AlignResult"
```

---

### Task 2: Parasail backend (factor existing proven logic)

**Files:**
- Create: `src/alignair/align/parasail.py`
- Modify: `src/alignair/align/__init__.py` (uncomment the `parasail` import)
- Test: `tests/alignair/align/test_parasail.py`

**Interfaces:**
- Consumes: `AlignResult` from Task 1.
- Produces: `ParasailAligner` implementing `Aligner`; module-level `parasail_available() -> bool`. Reuses the run-length CIGAR helper as `_ops(gapped_q, gapped_r) -> str` (identical to `io/alignment.py:_ops`).

- [ ] **Step 1: Write the failing test** (expected values computed against parasail `sg_dx_trace_striped_16`, match=2/mismatch=-1/gapopen=3/gapext=1)

```python
# tests/alignair/align/test_parasail.py
import pytest
from alignair.align.parasail import ParasailAligner, parasail_available

pytestmark = pytest.mark.skipif(not parasail_available(), reason="parasail not installed")
AL = ParasailAligner()


def test_identity():
    r = AL.align("ACGTACGT", "ACGTACGT")
    assert (r.cigar, r.t_start, r.t_end) == ("8M", 0, 8)
    assert r.q_start == 0 and r.q_end == 8 and r.score == 16


def test_germline_5p_trim_sets_t_start():
    r = AL.align("ACGTACGT", "GGGACGTACGT")     # 3 free germline bases on the 5' end
    assert (r.cigar, r.t_start, r.t_end) == ("8M", 3, 11)
    assert r.score == 16


def test_single_mismatch_is_M_with_lower_score():
    r = AL.align("ACGTACGT", "ACGTTCGT")
    assert (r.cigar, r.t_start, r.t_end) == ("8M", 0, 8)
    assert r.score == 13


def test_query_insertion():
    r = AL.align("ACGTAACGT", "ACGTACGT")        # extra query A -> insertion (query-only)
    assert (r.cigar, r.t_start, r.t_end) == ("4M1I4M", 0, 8)
    assert r.q_end == 9 and r.score == 13


def test_germline_deletion():
    r = AL.align("ACGTCGT", "ACGTACGT")          # missing query A -> deletion (germline-only)
    assert (r.cigar, r.t_start, r.t_end) == ("4M1D3M", 0, 8)
    assert r.q_end == 7 and r.score == 11


def test_too_short_or_empty_returns_none():
    assert AL.align("", "ACGTACGT") is None
    assert AL.align("ACGT", "") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/align/test_parasail.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'alignair.align.parasail'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/align/parasail.py
from __future__ import annotations
from .backend import AlignResult

_PARASAIL = None


def _parasail():
    global _PARASAIL
    if _PARASAIL is None:
        import parasail
        _PARASAIL = (parasail, parasail.matrix_create("ACGTN", 2, -1))
    return _PARASAIL


def parasail_available() -> bool:
    try:
        _parasail()
        return True
    except Exception:
        return False


def _ops(gapped_q: str, gapped_r: str) -> str:
    """Core CIGAR: both bases -> M, query gap -> D (germline-only), ref gap -> I (query-only)."""
    out, run_op, run_len = [], None, 0
    for q, r in zip(gapped_q, gapped_r):
        op = "D" if q == "-" else ("I" if r == "-" else "M")
        if op == run_op:
            run_len += 1
        else:
            if run_op:
                out.append(f"{run_len}{run_op}")
            run_op, run_len = op, 1
    if run_op:
        out.append(f"{run_len}{run_op}")
    return "".join(out)


class ParasailAligner:
    """Query-global / germline-ends-free gap-affine alignment (parasail sg_dx)."""

    def align(self, query: str, target: str) -> AlignResult | None:
        if len(query) < 1 or len(target) < 1:
            return None
        par, mat = _parasail()
        res = par.sg_dx_trace_striped_16(query, target, 3, 1, mat)
        q, r = res.traceback.query, res.traceback.ref
        cols = [i for i, c in enumerate(q) if c != "-"]
        if not cols:
            return None
        a, b = cols[0], cols[-1] + 1
        t_start = sum(1 for c in r[:a] if c != "-")
        cq, cr = q[a:b], r[a:b]
        t_end = t_start + sum(1 for c in cr if c != "-")
        return AlignResult(score=float(res.score), cigar=_ops(cq, cr),
                           q_start=0, q_end=len(query), t_start=t_start, t_end=t_end)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/align/test_parasail.py tests/alignair/align/test_backend.py -v`
Expected: all PASS (uncomment the `from .parasail import ParasailAligner` line in `__init__.py` first).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/align/parasail.py src/alignair/align/__init__.py tests/alignair/align/test_parasail.py
git commit -m "align: parasail backend (sg_dx, core CIGAR, ends-free germline)"
```

---

### Task 3: K-mer seed prefilter

**Files:**
- Create: `src/alignair/align/seed_prefilter.py`
- Modify: `src/alignair/align/__init__.py` (uncomment the `seed_prefilter` import)
- Test: `tests/alignair/align/test_seed_prefilter.py`

**Interfaces:**
- Produces: `SeedPrefilter(reference_set, k: int = 11)`; method `candidates(self, segment: str, gene: str, m: int, allowed: set[int] | None = None) -> list[int]` returning up to `m` allele indices ranked by shared-k-mer count (desc), restricted to `allowed` when given.
- Consumes (duck-typed reference): an object with `.gene(G)` returning an object with `.sequences: list[str]` and `.names: list[str]`. The real `ReferenceSet` (`src/alignair/reference/reference_set.py`) satisfies this; tests use a stub.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/align/test_seed_prefilter.py
from alignair.align.seed_prefilter import SeedPrefilter


class _Gene:
    def __init__(self, names, seqs):
        self.names, self.sequences = names, seqs


class _Ref:
    def __init__(self):
        self._g = {"V": _Gene(
            ["A*01", "A*02", "FAR*01"],
            ["ACGTACGTACGTACGTACGT",         # idx0
             "ACGTACGTACGTACGTACGA",         # idx1: 1 SNP from idx0
             "TTTTTGGGGGCCCCCAAAAA"])}        # idx2: unrelated
    def gene(self, g):
        return self._g[g]


def test_near_neighbor_segment_ranks_related_alleles_first():
    sp = SeedPrefilter(_Ref(), k=5)
    cand = sp.candidates("ACGTACGTACGTACGTACGT", "V", m=2)
    assert set(cand) == {0, 1}                    # the two related alleles, not the far one


def test_divergent_segment_admits_the_far_allele():
    # a read of the FAR allele must be admittable by raw k-mer match even though it shares
    # nothing with alleles 0/1 — this is the swap-robustness guarantee.
    sp = SeedPrefilter(_Ref(), k=5)
    cand = sp.candidates("TTTTTGGGGGCCCCCAAAAA", "V", m=1)
    assert cand == [2]


def test_allowed_restricts_to_genotype():
    sp = SeedPrefilter(_Ref(), k=5)
    cand = sp.candidates("ACGTACGTACGTACGTACGT", "V", m=3, allowed={1})
    assert cand == [1]


def test_segment_shorter_than_k_returns_empty():
    sp = SeedPrefilter(_Ref(), k=5)
    assert sp.candidates("AC", "V", m=3) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/align/test_seed_prefilter.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/align/seed_prefilter.py
from __future__ import annotations
from collections import defaultdict


def _kmers(seq: str, k: int):
    return (seq[i:i + k] for i in range(len(seq) - k + 1))


class SeedPrefilter:
    """Non-learned k-mer admission: rank candidate alleles by shared-k-mer count with a read
    segment, independent of the neural encoder. Lets a divergent/novel allele enter the
    alignment pool even when pooled-cosine retrieval misranks it."""

    def __init__(self, reference_set, k: int = 11):
        self.k = k
        self._index = {}          # gene -> {kmer: set(allele_idx)}
        for G in ("V", "D", "J"):
            try:
                gene = reference_set.gene(G)
            except Exception:
                continue
            idx = defaultdict(set)
            for ai, seq in enumerate(gene.sequences):
                for km in _kmers(seq.upper(), k):
                    idx[km].add(ai)
            self._index[G] = idx

    def candidates(self, segment: str, gene: str, m: int,
                   allowed: set[int] | None = None) -> list[int]:
        idx = self._index.get(gene.upper())
        if idx is None or len(segment) < self.k:
            return []
        counts = defaultdict(int)
        for km in _kmers(segment.upper(), self.k):
            for ai in idx.get(km, ()):
                if allowed is None or ai in allowed:
                    counts[ai] += 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [ai for ai, _ in ranked[:m]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/align/test_seed_prefilter.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/align/seed_prefilter.py src/alignair/align/__init__.py tests/alignair/align/test_seed_prefilter.py
git commit -m "align: non-learned k-mer seed prefilter (swap-robust candidate admission)"
```

---

### Task 4: Batched executor

**Files:**
- Create: `src/alignair/align/batch.py`
- Modify: `src/alignair/align/__init__.py` (uncomment the `batch` import)
- Test: `tests/alignair/align/test_batch.py`

**Interfaces:**
- Consumes: an `Aligner` (Task 1), `AlignResult`.
- Produces: `align_batch(pairs: list[tuple[str, str]], aligner, workers: int = 8) -> list[AlignResult | None]` — results in the **same order** as `pairs`; parasail/WFA release the GIL in C so a thread pool parallelizes.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/align/test_batch.py
import pytest
from alignair.align.batch import align_batch
from alignair.align.parasail import ParasailAligner, parasail_available

pytestmark = pytest.mark.skipif(not parasail_available(), reason="parasail not installed")


def test_batch_preserves_order_and_matches_serial():
    al = ParasailAligner()
    pairs = [("ACGTACGT", "ACGTACGT"), ("ACGTACGT", "ACGTTCGT"), ("ACGTCGT", "ACGTACGT")]
    got = align_batch(pairs, al, workers=2)
    exp = [al.align(q, t) for q, t in pairs]
    assert [r.cigar for r in got] == [r.cigar for r in exp]
    assert [r.score for r in got] == [r.score for r in exp]


def test_batch_handles_none_results():
    al = ParasailAligner()
    got = align_batch([("", "ACGT"), ("ACGTACGT", "ACGTACGT")], al, workers=2)
    assert got[0] is None and got[1] is not None


def test_empty_batch():
    assert align_batch([], ParasailAligner()) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/align/test_batch.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/align/batch.py
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from .backend import AlignResult


def align_batch(pairs, aligner, workers: int = 8) -> list[AlignResult | None]:
    """Align many (query, target) pairs, preserving input order. Threaded: the C aligners
    release the GIL, so this parallelizes across cores."""
    if not pairs:
        return []
    if workers <= 1 or len(pairs) == 1:
        return [aligner.align(q, t) for q, t in pairs]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda p: aligner.align(p[0], p[1]), pairs))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/align/ -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/align/batch.py src/alignair/align/__init__.py tests/alignair/align/test_batch.py
git commit -m "align: threaded order-preserving batch executor"
```

---

### Task 5: WFA2 backend (optional, behind the same protocol)

**Files:**
- Create: `src/alignair/align/wfa.py`
- Test: `tests/alignair/align/test_wfa.py`

**Interfaces:**
- Produces: `WFAAligner` implementing `Aligner` (same `AlignResult` contract as `ParasailAligner`); `wfa_available() -> bool`. `get_aligner("wfa")` (Task 1) returns it when available.

**Setup:** install pywfa into the venv:
```bash
.venv/bin/pip install pywfa
```
If the build fails (needs a C toolchain / WFA2-lib), this task is **deferred** — the package already works on parasail via `get_aligner()` fallback. Record the failure and move on; do not block.

**Correctness gate:** the WFA backend is correct iff it produces the **same allele ranking and the same germline coordinates** as parasail on the shared cases. The parity test (not hand-computed WFA internals) is the spec — WFA's raw score scale differs, so parity is on `t_start`/`t_end`/`cigar` and on *ranking order*, not on the absolute `score` value.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/align/test_wfa.py
import pytest
from alignair.align.parasail import ParasailAligner, parasail_available

wfa = pytest.importorskip("pywfa")
from alignair.align.wfa import WFAAligner, wfa_available

pytestmark = pytest.mark.skipif(not (wfa_available() and parasail_available()),
                                reason="pywfa or parasail unavailable")
CASES = [("ACGTACGT", "ACGTACGT"), ("ACGTACGT", "GGGACGTACGT"),
         ("ACGTACGT", "ACGTTCGT"), ("ACGTAACGT", "ACGTACGT"), ("ACGTCGT", "ACGTACGT")]


def test_wfa_coords_and_cigar_match_parasail():
    w, p = WFAAligner(), ParasailAligner()
    for q, t in CASES:
        rw, rp = w.align(q, t), p.align(q, t)
        assert (rw.cigar, rw.t_start, rw.t_end) == (rp.cigar, rp.t_start, rp.t_end), (q, t)


def test_wfa_ranking_matches_parasail_on_siblings():
    # the operative use: pick the best of several germlines. WFA and parasail must agree on argmax.
    w, p = WFAAligner(), ParasailAligner()
    seg = "ACGTACGTACGTACGT"
    germs = ["ACGTACGTACGTACGT", "ACGTACGTACGTACGA", "TTTTGGGGCCCCAAAA"]
    aw = [w.align(seg, g).score for g in germs]
    ap = [p.align(seg, g).score for g in germs]
    assert aw.index(max(aw)) == ap.index(max(ap)) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/align/test_wfa.py -v`
Expected: FAIL/SKIP. If pywfa installed: FAIL `ModuleNotFoundError: alignair.align.wfa`. If not installed: SKIP (acceptable terminal state for this task).

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/align/wfa.py
from __future__ import annotations
from .backend import AlignResult

_OK = None


def wfa_available() -> bool:
    global _OK
    if _OK is None:
        try:
            import pywfa  # noqa: F401
            _OK = True
        except Exception:
            _OK = False
    return _OK


def _cigar_from_tuples(tuples) -> tuple[str, int, int, int]:
    """pywfa cigartuples: list of (op_code, length). op 0=M,7=eq,8=diff -> M; 1=I (query); 2=D
    (germline). Returns (core_cigar, q_consumed, t_consumed_in_core, leading_t_skip)."""
    out, q_used, t_used = [], 0, 0
    run_op, run_len = None, 0
    for op, length in tuples:
        c = "I" if op == 1 else ("D" if op == 2 else "M")
        if c in ("M",):
            q_used += length; t_used += length
        elif c == "I":
            q_used += length
        elif c == "D":
            t_used += length
        if c == run_op:
            run_len += length
        else:
            if run_op:
                out.append(f"{run_len}{run_op}")
            run_op, run_len = c, length
    if run_op:
        out.append(f"{run_len}{run_op}")
    return "".join(out), q_used, t_used, 0


class WFAAligner:
    """WFA2 (pywfa) backend: query global, germline (text) ends free. Same AlignResult contract
    as ParasailAligner; correctness is enforced by parity tests against parasail, since WFA's
    raw score scale differs (higher-is-better is preserved)."""

    def __init__(self, match: int = 2, mismatch: int = 4, gap_open: int = 6, gap_ext: int = 2):
        from pywfa import WavefrontAligner
        # text (germline) ends free so 5'/3' germline trim is not penalized; pattern (query) global.
        self._mk = lambda q: WavefrontAligner(
            q, span="ends-free", pattern_begin_free=0, pattern_end_free=0,
            text_begin_free=len(q) * 0, text_end_free=10_000,
            scope="full", match=match, mismatch=mismatch,
            gap_opening=gap_open, gap_extension=gap_ext)

    def align(self, query: str, target: str) -> AlignResult | None:
        if len(query) < 1 or len(target) < 1:
            return None
        aligner = self._mk(query)
        res = aligner(target)                                # align text(germline) to pattern(query)
        if res.cigartuples is None:
            return None
        cigar, q_used, t_core, _ = _cigar_from_tuples(res.cigartuples)
        t_start = int(getattr(res, "text_start", 0) or 0)
        return AlignResult(score=float(res.score), cigar=cigar, q_start=0, q_end=len(query),
                           t_start=t_start, t_end=t_start + t_core)
```

Note: the exact `pywfa.WavefrontAligner` keyword names and the sign of `res.score` must be verified against the installed pywfa version — adjust the constructor kwargs and, if pywfa reports score as a *penalty* (lower-is-better), negate it in the `AlignResult` so higher-is-better holds. **The two parity tests are the acceptance gate**; iterate the constructor until they pass.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/align/test_wfa.py -v`
Expected: PASS if pywfa installed; otherwise SKIP (deferred — package still ships on parasail).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/align/wfa.py tests/alignair/align/test_wfa.py
git commit -m "align: WFA2 backend (pywfa) behind the aligner protocol, parity-tested vs parasail"
```

---

## Done criteria

- `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/align/ -v` is green (WFA tests pass or skip).
- `get_aligner()` returns WFA when pywfa is present, parasail otherwise, both satisfying the same `AlignResult` contract.
- `SeedPrefilter` admits a divergent allele a retrieval-miss would drop (the swap-robustness unit).
- No torch/model/GPU import anywhere under `src/alignair/align/`.

## Follow-on plans (not this plan)

1. **`predict_reads` rewrite** — union pool (retrieval top-k ∪ seed prefilter) → `align_batch` → pick + score-band equivalence set + coords/CIGAR; delete the differentiable-DP reader branch; extend `tests/alignair/inference/test_dnalignair_infer.py`.
2. **Training simplification + retrain** — strip band/DP losses, add the per-position contrastive term, `scripts/train_fast.py`; gates = set-aware recall + assay + embargo.
3. **Encoder acceleration** (Phase 2/3) — fp16 + `torch.compile`; ONNX/TensorRT export.
