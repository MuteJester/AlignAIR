# Amino-Acid AlignAIR — Phase 0 (Information Audit + Semantic Prototype) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the amino-acid semantic-prototype primitives (translation, codon-provenance projection, AA tokenizer, translated germlines with anchor QC, protein-equivalence classes) and produce a **frozen Phase-0 diagnostic report** that answers the go/no-go identifiability gate — with **no neural training**.

**Architecture:** A new `src/alignair/aa/` subpackage of pure, unit-tested primitives, consumed by a simulation-driven `build_phase0_report()` that streams GenAIRR IGH records, projects nucleotide segment provenance onto amino-acid residues, and audits how much V/J allele identity and D-segment signal survive translation. The nucleotide package is untouched. No model, no loss, no training in this phase.

**Tech Stack:** Python 3, NumPy, PyTorch (tensors only, for the tokenizer batch helper), GenAIRR 2.2.0 (`HUMAN_IGH_OGRDB` dataconfig + `Experiment.stream_records`), pytest.

## Global Constraints

- **Branch:** all work on `experiment/aminoacid-alignair` (already created off `origin/main`, upstream unset). Never commit to `main`.
- **No neural training in Phase 0.** No model instantiation, no loss, no optimizer.
- **Half-open `[start, end)` coordinate semantics everywhere** — segment ranges, AA bounds, windows.
- **Determinism:** every simulation call takes an explicit `seed`. No wall-clock/random seeding.
- **Pad is token id 0 with NO character** in the AA vocab (`P` is proline, not pad).
- **Stop `*` and unknown `X` are distinct token ids** — never made equivalent.
- **Commits:** conventional-commit messages; **never** add a `Co-Authored-By: Claude` line or any Claude/Anthropic mention (repo convention).
- **Run environment:** `PYTHONPATH=src .venv/bin/python …` (package is not pip-installed). Tests: `PYTHONPATH=src .venv/bin/python -m pytest …`.
- **Tests:** live under `tests/alignair/aa/` with **no `__init__.py`** (repo uses `--import-mode=importlib`). Any test that loads a real GenAIRR dataconfig or runs simulation is marked `@pytest.mark.slow`.
- **GenAIRR facts (verified):** allele objects via `dc.allele_list("v"|"d"|"j")`; attrs `.name`, `.ungapped_seq` (lowercase nt), `.gapped_seq`, `.anchor` (0-based ungapped index of the first base of the conserved codon; `int` for V/J, `None` for D). `from GenAIRR.utilities import translate`. Simulate: `GenAIRR.Experiment.on("HUMAN_IGH_OGRDB").recombine().mutate(model="s5f", rate=0.05).stream_records(n=…, seed=…)` yields a dict with `sequence`, `sequence_aa`, `junction_aa`, `v/d/j_sequence_start/end`, `junction_start`, `productive`, `vj_in_frame`, `stop_codon`, `v/d/j_call`. **There is no frame field** — derive `frame_offset = junction_start % 3` (valid only when `junction_start is not None`).

---

### Task 1: Package scaffold + codon-provenance projection (`aa/translation.py`)

The highest-risk primitive: mapping nucleotide segment provenance onto amino-acid residues, masking codons that straddle a segment boundary. This is the "coordinate projection is not `//3`" logic from the spec (§5.2).

**Files:**
- Create: `src/alignair/aa/__init__.py`
- Create: `src/alignair/aa/translation.py`
- Test: `tests/alignair/aa/test_translation.py`

**Interfaces:**
- Produces:
  - `CODON_TABLE: dict[str, str]` — 64 DNA codons → 1-letter AA (`*` for stops).
  - `translate_nt(nt: str, frame: int = 0, unknown: str = "X", stop: str = "*") -> str` — translate from `frame`, dropping the trailing partial codon; non-ACGT codon → `unknown`.
  - `frame_offset_of(junction_start: int | None) -> int | None` — `junction_start % 3`, or `None`.
  - `@dataclass(frozen=True) AAProjection` with fields `n_residues: int`, `provenance: tuple[str | None, ...]`, `aa_bounds: dict[str, tuple[int, int] | None]`, `phase: dict[str, dict[str, int]]`.
  - `project_codons(seq_len_nt: int, frame_offset: int, segments: dict[str, tuple[int, int]]) -> AAProjection`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_translation.py
import pytest
from alignair.aa.translation import (
    CODON_TABLE, translate_nt, frame_offset_of, project_codons, AAProjection,
)


def test_codon_table_has_64_entries_and_known_codons():
    assert len(CODON_TABLE) == 64
    assert CODON_TABLE["ATG"] == "M"
    assert CODON_TABLE["TGG"] == "W"
    assert CODON_TABLE["TGT"] == "C" and CODON_TABLE["TGC"] == "C"
    assert CODON_TABLE["TAA"] == CODON_TABLE["TAG"] == CODON_TABLE["TGA"] == "*"


def test_translate_nt_basic_frame_unknown_and_partial():
    assert translate_nt("ATGTGG") == "MW"
    assert translate_nt("TGGTAG") == "W*"
    assert translate_nt("AATGTGG", frame=1) == "MW"      # skip 1 base, then ATG TGG
    assert translate_nt("ATGA") == "M"                    # trailing partial dropped
    assert translate_nt("ATGNNN") == "MX"                 # non-ACGT codon -> X
    assert translate_nt("atgtgg") == "MW"                 # lowercase ok


def test_frame_offset_of():
    assert frame_offset_of(285) == 0
    assert frame_offset_of(286) == 1
    assert frame_offset_of(287) == 2
    assert frame_offset_of(None) is None


def test_project_codons_clean_vj_no_boundary_cut():
    # frame 0, V=[0,6) J=[6,12): residues r0,r1 in V ; r2,r3 in J ; no straddle.
    proj = project_codons(12, 0, {"V": (0, 6), "J": (6, 12)})
    assert proj.n_residues == 4
    assert proj.provenance == ("V", "V", "J", "J")
    assert proj.aa_bounds["V"] == (0, 2) and proj.aa_bounds["J"] == (2, 4)
    assert proj.phase["V"] == {"start": 0, "end": 0}
    assert proj.phase["J"] == {"start": 0, "end": 0}


def test_project_codons_boundary_straddle_is_masked():
    # frame 0, V=[0,5) J=[5,12): residue r1 spans nt[3,6) crossing the 5-boundary -> None (masked).
    proj = project_codons(12, 0, {"V": (0, 5), "J": (5, 12)})
    assert proj.provenance == ("V", None, "J", "J")
    assert proj.aa_bounds["V"] == (0, 1)          # only r0 fully owned by V
    assert proj.aa_bounds["J"] == (2, 4)
    assert proj.phase["V"]["end"] == 2            # 5 % 3 == 2 -> V/np boundary cuts a codon


def test_project_codons_frame_offset_shifts_residue_grid():
    # frame 1: residue r0 spans nt[1,4). V=[1,7) J=[7,13).
    proj = project_codons(13, 1, {"V": (1, 7), "J": (7, 13)})
    assert proj.n_residues == 4                    # (13 - 1)//3 == 4
    assert proj.provenance == ("V", "V", "J", "J")
    assert proj.aa_bounds["V"] == (0, 2)


def test_project_codons_d_region_and_np_gaps():
    # V=[0,6) np1=[6,9) D=[9,15) np2=[15,18) J=[18,24), frame 0.
    proj = project_codons(24, 0, {"V": (0, 6), "D": (9, 15), "J": (18, 24)})
    # residues: r0,r1 V ; r2 nt[6,9) np1 -> None ; r3 D ; r4 D ; r5 nt[15,18) np2 -> None ; r6,r7 J
    assert proj.provenance == ("V", "V", None, "D", "D", None, "J", "J")
    assert proj.aa_bounds["D"] == (3, 5)           # 2 fully-owned D residues


@pytest.mark.parametrize("fo", [0, 1, 2])
@pytest.mark.parametrize("vend", [6, 7, 8])
def test_project_codons_phase_matches_boundary_offset(fo, vend):
    proj = project_codons(30, fo, {"V": (fo, vend), "J": (vend, 30)})
    assert proj.phase["V"]["end"] == (vend - fo) % 3
    # a residue is owned by V iff its whole codon is inside [fo, vend)
    for r, owner in enumerate(proj.provenance):
        a = fo + 3 * r
        if a + 3 <= vend:
            assert owner == "V"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_translation.py -q`
Expected: FAIL / collection error — `ModuleNotFoundError: No module named 'alignair.aa'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/__init__.py
"""Amino-acid AlignAIR — experimental subpackage (Phase 0: audit + semantic prototype)."""
```

```python
# src/alignair/aa/translation.py
"""Amino-acid translation + codon-provenance projection.

Coordinate projection is NOT ``//3``: GenAIRR translates the assembled read in a
junction-derived reading frame and drops incomplete boundary codons. A residue is
assigned to a nucleotide segment only when ALL THREE of its codon's nucleotides lie
inside that segment; codons that straddle a segment boundary (or land in an N/P gap)
are masked (provenance ``None``).
"""
from __future__ import annotations

from dataclasses import dataclass

# Standard genetic code (NCBI transl_table=1). Codons enumerated with base order T,C,A,G
# so they align with the canonical amino-acid string below.
_BASES = "TCAG"
_AAS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE: dict[str, str] = {
    x + y + z: aa
    for (x, y, z), aa in zip(
        ((a, b, c) for a in _BASES for b in _BASES for c in _BASES), _AAS
    )
}


def translate_nt(nt: str, frame: int = 0, unknown: str = "X", stop: str = "*") -> str:
    """Translate ``nt`` from ``frame``; trailing partial codon dropped; non-ACGT codon -> ``unknown``."""
    s = nt[frame:].upper().replace("U", "T")
    out = []
    for i in range(0, len(s) - 2, 3):
        aa = CODON_TABLE.get(s[i:i + 3], unknown)
        out.append(stop if aa == "*" else aa)
    return "".join(out)


def frame_offset_of(junction_start: int | None) -> int | None:
    """Reading-frame offset of the assembled read = ``junction_start % 3`` (V-anchor frame)."""
    return None if junction_start is None else junction_start % 3


@dataclass(frozen=True)
class AAProjection:
    n_residues: int
    provenance: tuple                     # per residue: "V"/"D"/"J"/None (None = masked)
    aa_bounds: dict                       # seg -> (aa_start, aa_end) over fully-owned residues, or None
    phase: dict                           # seg -> {"start": 0|1|2, "end": 0|1|2}


def project_codons(seq_len_nt: int, frame_offset: int,
                   segments: dict) -> AAProjection:
    """Assign each residue of the in-frame translation a segment provenance.

    ``segments`` maps seg name -> half-open ``(start, end)`` nt coords in the assembled read.
    """
    n_res = max(0, (seq_len_nt - frame_offset) // 3)
    prov = []
    for r in range(n_res):
        a = frame_offset + 3 * r
        b = a + 3
        owner = None
        for seg, (s, e) in segments.items():
            if s <= a and b <= e:
                owner = seg
                break
        prov.append(owner)
    aa_bounds = {}
    for seg in segments:
        idx = [r for r, o in enumerate(prov) if o == seg]
        aa_bounds[seg] = (idx[0], idx[-1] + 1) if idx else None
    phase = {seg: {"start": (s - frame_offset) % 3, "end": (e - frame_offset) % 3}
             for seg, (s, e) in segments.items()}
    return AAProjection(n_res, tuple(prov), aa_bounds, phase)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_translation.py -q`
Expected: PASS (all tests green).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/__init__.py src/alignair/aa/translation.py tests/alignair/aa/test_translation.py
git commit -m "feat(aa): codon-provenance projection + AA translation primitives"
```

---

### Task 2: AA vocabulary + right-padding tokenizer (`aa/tokenizer.py`)

Realizes the §5.4 vocabulary/invalid-input policy: PAD(0, no char) + 20 residues + `X` + `*`, right-padding, explicit invalid-residue handling. Stop and unknown are distinct.

**Files:**
- Create: `src/alignair/aa/tokenizer.py`
- Test: `tests/alignair/aa/test_tokenizer_aa.py`

**Interfaces:**
- Produces:
  - `AA_TOKEN_DICT: dict[str, int]` (20 residues → 1..20, `X` → 21, `*` → 22).
  - `AA_VOCAB_SIZE: int == 23` (0 pad + 22).
  - `class InvalidResidueError(ValueError)`.
  - `class AAProteinTokenizer(token_dict=None, on_invalid="X")` with `.encode(seq: str) -> np.ndarray[int64]`.
  - `pad_tokenize_aa(sequences, token_dict=None, on_invalid="X") -> (LongTensor (B,Lmax), BoolTensor (B,Lmax))`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_tokenizer_aa.py
import numpy as np
import pytest
from alignair.aa.tokenizer import (
    AA_TOKEN_DICT, AA_VOCAB_SIZE, InvalidResidueError,
    AAProteinTokenizer, pad_tokenize_aa,
)


def test_vocab_shape_and_proline_gotcha():
    assert AA_VOCAB_SIZE == 23
    # 20 canonical residues -> 1..20, X -> 21, * -> 22 ; pad id 0 has NO character.
    assert AA_TOKEN_DICT["A"] == 1
    assert AA_TOKEN_DICT["P"] == 13          # proline is a real residue, NOT pad
    assert AA_TOKEN_DICT["Y"] == 20
    assert AA_TOKEN_DICT["X"] == 21
    assert AA_TOKEN_DICT["*"] == 22
    assert 0 not in AA_TOKEN_DICT.values()   # 0 is pad-only


def test_stop_and_unknown_are_distinct():
    assert AA_TOKEN_DICT["*"] != AA_TOKEN_DICT["X"]


def test_encode_maps_and_lowercases():
    tok = AAProteinTokenizer()
    np.testing.assert_array_equal(tok.encode("acdef"), tok.encode("ACDEF"))
    assert tok.encode("ACDEF").tolist() == [1, 2, 3, 4, 5]


def test_encode_invalid_maps_to_X_by_default():
    tok = AAProteinTokenizer()                      # on_invalid="X"
    # B, Z, J, U, O, gap and digit are non-canonical -> X (21)
    assert tok.encode("BZJUO-1").tolist() == [21] * 7


def test_encode_invalid_raises_when_configured():
    tok = AAProteinTokenizer(on_invalid="raise")
    with pytest.raises(InvalidResidueError):
        tok.encode("AB")                            # B invalid


def test_pad_tokenize_aa_right_pads_and_masks():
    tokens, mask = pad_tokenize_aa(["AC", "ACDE"])
    assert tokens.shape == (2, 4) and mask.shape == (2, 4)
    assert tokens[0].tolist() == [1, 2, 0, 0]       # right pad with 0
    assert mask[0].tolist() == [True, True, False, False]
    assert mask[1].tolist() == [True, True, True, True]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_tokenizer_aa.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.aa.tokenizer'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/tokenizer.py
"""Amino-acid tokenizer. PAD is id 0 with NO character ('P' is proline). Right-padding.

Stop ``*`` and unknown ``X`` are distinct tokens. Stop-rejection for input reads is an
input-contract concern enforced at the pipeline layer (Phase 2), not here.
"""
from __future__ import annotations

import numpy as np
import torch

_CANONICAL = "ACDEFGHIKLMNPQRSTVWY"          # 20 residues, alphabetical by 1-letter code
AA_TOKEN_DICT: dict[str, int] = {aa: i + 1 for i, aa in enumerate(_CANONICAL)}
AA_TOKEN_DICT["X"] = 21
AA_TOKEN_DICT["*"] = 22
AA_VOCAB_SIZE = 23                            # 0 pad + 20 residues + X + *


class InvalidResidueError(ValueError):
    """Raised by the tokenizer when on_invalid='raise' and a non-vocab residue is seen."""


class AAProteinTokenizer:
    """Encode a protein string to int ids. ``on_invalid``: 'X' maps unknown residues to X; 'raise' errors."""

    def __init__(self, token_dict: dict | None = None, on_invalid: str = "X"):
        if on_invalid not in ("X", "raise"):
            raise ValueError(f"on_invalid must be 'X' or 'raise', got {on_invalid!r}")
        self.token_dict = token_dict or dict(AA_TOKEN_DICT)
        self.on_invalid = on_invalid
        self._x = self.token_dict["X"]

    def encode(self, sequence: str) -> np.ndarray:
        out = []
        for ch in sequence.upper():
            tok = self.token_dict.get(ch)
            if tok is not None:
                out.append(tok)
            elif self.on_invalid == "raise":
                raise InvalidResidueError(f"invalid residue {ch!r} in {sequence!r}")
            else:
                out.append(self._x)
        return np.array(out, dtype=np.int64)


def pad_tokenize_aa(sequences, token_dict: dict | None = None, on_invalid: str = "X"):
    """Right-pad a batch of protein strings. Returns (LongTensor (B,Lmax), BoolTensor mask True=valid)."""
    tok = AAProteinTokenizer(token_dict, on_invalid)
    encoded = [tok.encode(s) for s in sequences]
    lmax = max((len(e) for e in encoded), default=0)
    tokens = torch.zeros(len(encoded), lmax, dtype=torch.long)
    mask = torch.zeros(len(encoded), lmax, dtype=torch.bool)
    for i, e in enumerate(encoded):
        if len(e):
            tokens[i, :len(e)] = torch.tensor(e, dtype=torch.long)
            mask[i, :len(e)] = True
    return tokens, mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_tokenizer_aa.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/tokenizer.py tests/alignair/aa/test_tokenizer_aa.py
git commit -m "feat(aa): protein tokenizer with explicit vocab + invalid-residue policy"
```

---

### Task 3: Germline translation + conserved-anchor QC (`aa/reference.py`)

Translate V/J germlines in their anchor frame and **quarantine** any whose translation lacks the conserved anchor residue (Cys-104 for V, Trp-118 for J) — the §5.1 germline-QC step. D is deliberately excluded (no canonical protein form, §5.3).

**Files:**
- Create: `src/alignair/aa/reference.py`
- Test: `tests/alignair/aa/test_reference_aa.py`

**Interfaces:**
- Consumes: `translate_nt` from `aa/translation.py`.
- Produces:
  - `@dataclass(frozen=True) AAGermline(name: str, aa_seq: str, frame: int, anchor_aa: int, gene: str)`.
  - `@dataclass AAGermlineSet(genes: dict[str, dict[str, AAGermline]], quarantined: dict[str, dict[str, str]])`.
  - `ANCHOR_AA: dict[str, str]` = `{"V": "C", "J": "W"}`.
  - `translate_germline(name, ungapped_seq, anchor, gene) -> tuple[AAGermline | None, str | None]` (germline, reason-if-quarantined).
  - `build_aa_germlines(dc, genes=("V", "J")) -> AAGermlineSet`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_reference_aa.py
import pytest
from alignair.aa.reference import (
    ANCHOR_AA, AAGermline, translate_germline, build_aa_germlines,
)


def test_translate_germline_passes_anchor_qc():
    # frame 0, Cys (TGT) codon at nt index 6 -> anchor 6, residue index 2.
    seq = "ATGAAATGTGGG"          # M K C G
    germ, reason = translate_germline("IGHV-test*01", seq, anchor=6, gene="V")
    assert reason is None
    assert isinstance(germ, AAGermline)
    assert germ.aa_seq == "MKCG"
    assert germ.frame == 0
    assert germ.anchor_aa == 2
    assert germ.aa_seq[germ.anchor_aa] == ANCHOR_AA["V"]


def test_translate_germline_quarantines_wrong_anchor_residue():
    # anchor points at GGG (Gly), not Cys -> quarantined.
    seq = "ATGAAAGGGTGG"          # M K G W
    germ, reason = translate_germline("IGHV-bad*01", seq, anchor=6, gene="V")
    assert germ is None
    assert reason is not None and "C" in reason


def test_translate_germline_respects_nonzero_frame():
    # frame 2 (anchor 8 -> 8 % 3 == 2). Translate from base 2: TGT GGG -> C G ; anchor residue 0.
    seq = "AATGTGGG"              # bases[2:] = TGTGGG -> C G
    germ, reason = translate_germline("IGHV-f2*01", seq, anchor=8, gene="V")
    # anchor 8 is beyond this toy seq; use a real-length check instead:
    # rebuild a frame-2 example that is in range:
    seq2 = "AATGTGGGAAA"          # bases[2:] = TGTGGGAAA -> C G K ; anchor 2 -> residue 0
    germ2, reason2 = translate_germline("IGHV-f2b*01", seq2, anchor=2, gene="V")
    assert reason2 is None and germ2.frame == 2 and germ2.anchor_aa == 0
    assert germ2.aa_seq[0] == "C"


@pytest.mark.slow
def test_build_aa_germlines_real_igh_all_v_and_j_pass_anchor_qc():
    import GenAIRR as ga
    dc = ga.HUMAN_IGH_OGRDB
    gset = build_aa_germlines(dc, genes=("V", "J"))
    # every V allele translation carries Cys at its anchor residue; every J carries Trp.
    assert len(gset.genes["V"]) >= 190          # 198 V alleles, near-all pass
    assert len(gset.genes["J"]) == 7            # all 7 J pass
    for germ in gset.genes["V"].values():
        assert germ.aa_seq[germ.anchor_aa] == "C"
    for germ in gset.genes["J"].values():
        assert germ.aa_seq[germ.anchor_aa] == "W"
    # quarantine, if any, is small and reported with a reason string.
    assert all(isinstance(r, str) for r in gset.quarantined["V"].values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_reference_aa.py -q -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.aa.reference'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/reference.py
"""Translated V/J germlines with conserved-anchor QC.

D is intentionally excluded: a D allele has no single canonical protein form (it appears
in any frame, is trimmed both ends, and its boundary codons absorb N-region nucleotides).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .translation import translate_nt

ANCHOR_AA: dict[str, str] = {"V": "C", "J": "W"}   # Cys-104 (V), Trp-118 (J)


@dataclass(frozen=True)
class AAGermline:
    name: str
    aa_seq: str        # in-frame translation of the ungapped germline
    frame: int         # anchor % 3
    anchor_aa: int     # residue index of the conserved anchor
    gene: str          # "V" | "J"


@dataclass
class AAGermlineSet:
    genes: dict = field(default_factory=dict)         # gene -> {name: AAGermline}
    quarantined: dict = field(default_factory=dict)   # gene -> {name: reason}


def translate_germline(name: str, ungapped_seq: str, anchor: int, gene: str):
    """Translate one germline in its anchor frame; QC the conserved anchor residue.

    Returns ``(AAGermline, None)`` on pass, or ``(None, reason)`` if the anchor residue is
    absent/wrong (bad frame or pseudogene) -> quarantine.
    """
    expected = ANCHOR_AA[gene]
    frame = anchor % 3
    aa = translate_nt(ungapped_seq, frame)
    anchor_aa = (anchor - frame) // 3
    if anchor_aa >= len(aa):
        return None, f"anchor residue index {anchor_aa} out of range (len {len(aa)})"
    if aa[anchor_aa] != expected:
        return None, f"anchor residue {aa[anchor_aa]!r} != expected {expected!r}"
    return AAGermline(name, aa, frame, anchor_aa, gene), None


def build_aa_germlines(dc, genes=("V", "J")) -> AAGermlineSet:
    """Build translated V/J germlines from a GenAIRR dataconfig, quarantining anchor-QC failures."""
    out: dict[str, dict] = {}
    quar: dict[str, dict] = {}
    for g in genes:
        gU = g.upper()
        out[gU], quar[gU] = {}, {}
        for allele in dc.allele_list(g.lower()):
            if allele.anchor is None:
                quar[gU][allele.name] = "no conserved anchor"
                continue
            germ, reason = translate_germline(
                allele.name, allele.ungapped_seq.upper(), int(allele.anchor), gU)
            if germ is None:
                quar[gU][allele.name] = reason
            else:
                out[gU][allele.name] = germ
    return AAGermlineSet(out, quar)
```

- [ ] **Step 4: Run tests to verify they pass**

Run (fast): `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_reference_aa.py -q -m "not slow"`
Expected: PASS.
Run (slow, real dataconfig): `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_reference_aa.py -q -m slow`
Expected: PASS (all V carry Cys, all J carry Trp at their anchors).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/reference.py tests/alignair/aa/test_reference_aa.py
git commit -m "feat(aa): translated V/J germlines with conserved-anchor QC"
```

---

### Task 4: Protein-equivalence classes + DB-version-stamped class map (`aa/equivalence.py`)

Group alleles whose in-frame translation is identical into equivalence classes, the object the model will predict instead of nucleotide alleles (§3). Stamp the map with a germline-DB hash and support ambiguity-set expansion + JSON round-trip.

**Files:**
- Create: `src/alignair/aa/equivalence.py`
- Test: `tests/alignair/aa/test_equivalence.py`

**Interfaces:**
- Produces:
  - `db_hash(name_seq_pairs: Iterable[tuple[str, str]]) -> str` (16-hex sha1 of sorted `(name, seq)`).
  - `@dataclass EquivalenceClassMap(gene, db_hash, allele_class: dict[str,int], class_members: dict[int, list[str]])` with `.class_of(name)`, `.ambiguity_set(class_id)`, `.n_classes()`, `.to_json()`, `EquivalenceClassMap.from_json(d)`.
  - `build_equivalence_classes(gene: str, aa_by_name: dict[str, str], db_hash: str) -> EquivalenceClassMap`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_equivalence.py
from alignair.aa.equivalence import (
    db_hash, build_equivalence_classes, EquivalenceClassMap,
)


def _toy():
    # a*01 and a*02 translate identically (synonymous) ; b*01 differs.
    aa = {"a*01": "MKV", "a*02": "MKV", "b*01": "MKL"}
    return build_equivalence_classes("V", aa, db_hash([("a*01", "atg"), ("b*01", "atc")]))


def test_identical_translations_share_a_class():
    cmap = _toy()
    assert cmap.n_classes() == 2
    assert cmap.class_of("a*01") == cmap.class_of("a*02")
    assert cmap.class_of("b*01") != cmap.class_of("a*01")


def test_ambiguity_set_returns_all_alleles_in_class():
    cmap = _toy()
    cid = cmap.class_of("a*01")
    assert cmap.ambiguity_set(cid) == ["a*01", "a*02"]


def test_json_round_trip_is_identity():
    cmap = _toy()
    back = EquivalenceClassMap.from_json(cmap.to_json())
    assert back.gene == cmap.gene
    assert back.db_hash == cmap.db_hash
    assert back.allele_class == cmap.allele_class
    assert back.class_members == cmap.class_members


def test_db_hash_is_deterministic_and_sensitive():
    h1 = db_hash([("a*01", "atg"), ("b*01", "atc")])
    h2 = db_hash([("b*01", "atc"), ("a*01", "atg")])   # order-independent
    h3 = db_hash([("a*01", "atg"), ("b*01", "atg")])   # a seq changed
    assert h1 == h2
    assert h1 != h3 and len(h1) == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_equivalence.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.aa.equivalence'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/equivalence.py
"""Protein-equivalence classes: alleles with identical in-frame translation collapse to one
class (they are indistinguishable in protein space). The model predicts the class; inference
expands it back to the nucleotide-allele ambiguity set. Stamped with a germline-DB hash."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable


def db_hash(name_seq_pairs: Iterable[tuple]) -> str:
    """Order-independent 16-hex digest of the germline DB (sorted (name, seq) pairs)."""
    h = hashlib.sha1()
    for name, seq in sorted(name_seq_pairs):
        h.update(str(name).encode()); h.update(b"\0")
        h.update(str(seq).upper().encode()); h.update(b"\n")
    return h.hexdigest()[:16]


@dataclass
class EquivalenceClassMap:
    gene: str
    db_hash: str
    allele_class: dict          # allele name -> class id (int)
    class_members: dict         # class id (int) -> sorted [allele names]

    def class_of(self, name: str) -> int:
        return self.allele_class[name]

    def ambiguity_set(self, class_id: int) -> list:
        return list(self.class_members[class_id])

    def n_classes(self) -> int:
        return len(self.class_members)

    def to_json(self) -> dict:
        return {
            "schema": "alignair.aa.equivalence.v1",
            "gene": self.gene,
            "db_hash": self.db_hash,
            "allele_class": dict(self.allele_class),
            "class_members": {str(k): list(v) for k, v in self.class_members.items()},
        }

    @classmethod
    def from_json(cls, d: dict) -> "EquivalenceClassMap":
        return cls(
            gene=d["gene"],
            db_hash=d["db_hash"],
            allele_class=dict(d["allele_class"]),
            class_members={int(k): list(v) for k, v in d["class_members"].items()},
        )


def build_equivalence_classes(gene: str, aa_by_name: dict, db_hash: str) -> EquivalenceClassMap:
    """Assign a class id per allele; alleles sharing an identical translation share a class.

    Class ids are assigned in order of first appearance over ``sorted(aa_by_name)`` for
    determinism.
    """
    by_seq: dict[str, int] = {}
    members: dict[int, list] = {}
    allele_class: dict[str, int] = {}
    for name in sorted(aa_by_name):
        aa = aa_by_name[name]
        cid = by_seq.setdefault(aa, len(by_seq))
        members.setdefault(cid, []).append(name)
        allele_class[name] = cid
    return EquivalenceClassMap(gene, db_hash, allele_class,
                               {cid: sorted(m) for cid, m in members.items()})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_equivalence.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/equivalence.py tests/alignair/aa/test_equivalence.py
git commit -m "feat(aa): protein-equivalence classes with DB-stamped class map"
```

---

### Task 5: Identifiability + coverage-conditioned audit primitives (`aa/audit.py`)

The metrics behind the Phase-0A go/no-go gate (§3): full-length identifiability from a class map, plus the coverage-aware version (a truncated read that sees only a window of V collapses classes further).

**Files:**
- Create: `src/alignair/aa/audit.py`
- Test: `tests/alignair/aa/test_audit.py`

**Interfaces:**
- Consumes: `EquivalenceClassMap` from `aa/equivalence.py`.
- Produces:
  - `identifiability_summary(cmap: EquivalenceClassMap) -> dict` with keys `gene, n_alleles, n_classes, n_unique_alleles, frac_alleles_uniquely_identifiable, ambiguity_size_hist, max_ambiguity`.
  - `window_identifiability(aa_by_name: dict[str, str], lo: int, hi: int) -> dict` with keys `lo, hi, n_alleles, n_classes, frac_uniquely_identifiable`.
  - `prefix_identifiability_curve(aa_by_name: dict[str, str], lengths: list[int]) -> list[dict]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_audit.py
from alignair.aa.equivalence import build_equivalence_classes, db_hash
from alignair.aa.audit import (
    identifiability_summary, window_identifiability, prefix_identifiability_curve,
)


def test_identifiability_summary_counts_classes_and_unique():
    aa = {"a*01": "MKV", "a*02": "MKV", "b*01": "MKL"}
    cmap = build_equivalence_classes("V", aa, db_hash([("a*01", "x")]))
    s = identifiability_summary(cmap)
    assert s["gene"] == "V"
    assert s["n_alleles"] == 3
    assert s["n_classes"] == 2
    assert s["n_unique_alleles"] == 1                        # only b*01 is alone
    assert abs(s["frac_alleles_uniquely_identifiable"] - 1 / 3) < 1e-9
    assert s["ambiguity_size_hist"] == {"1": 1, "2": 1}      # one class of 1, one of 2
    assert s["max_ambiguity"] == 2


def test_window_identifiability_collapses_on_short_window():
    aa = {"a": "MKVW", "b": "MKLW", "c": "MKVW"}
    # window [0,2) = "MK" for all -> 1 class, nothing unique.
    w = window_identifiability(aa, 0, 2)
    assert w["n_classes"] == 1 and w["frac_uniquely_identifiable"] == 0.0
    # full window: a==c collide, b alone -> 2 classes, only b unique.
    full = window_identifiability(aa, 0, 4)
    assert full["n_classes"] == 2
    assert abs(full["frac_uniquely_identifiable"] - 1 / 3) < 1e-9


def test_prefix_curve_is_monotonic_nondecreasing_in_classes():
    aa = {"a": "MKVW", "b": "MKLW", "c": "MQVW"}
    curve = prefix_identifiability_curve(aa, [1, 2, 3, 4])
    classes = [pt["n_classes"] for pt in curve]
    assert classes == sorted(classes)                        # more residues never merges classes
    assert curve[0]["lo"] == 0 and curve[0]["hi"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_audit.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.aa.audit'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/audit.py
"""Identifiability audit: how many alleles survive translation as uniquely distinguishable,
full-length and under coverage-limited windows (truncated reads collapse classes further)."""
from __future__ import annotations

from .equivalence import EquivalenceClassMap


def identifiability_summary(cmap: EquivalenceClassMap) -> dict:
    sizes = [len(m) for m in cmap.class_members.values()]
    n_alleles = len(cmap.allele_class)
    n_unique = sum(1 for s in sizes if s == 1)
    hist: dict[int, int] = {}
    for s in sizes:
        hist[s] = hist.get(s, 0) + 1
    return {
        "gene": cmap.gene,
        "n_alleles": n_alleles,
        "n_classes": len(sizes),
        "n_unique_alleles": n_unique,
        "frac_alleles_uniquely_identifiable": (n_unique / n_alleles) if n_alleles else 0.0,
        "ambiguity_size_hist": {str(k): hist[k] for k in sorted(hist)},
        "max_ambiguity": max(sizes) if sizes else 0,
    }


def window_identifiability(aa_by_name: dict, lo: int, hi: int) -> dict:
    """Group alleles by ``aa_seq[lo:hi]``; a covered sub-window merges classes that agree on it."""
    groups: dict[str, list] = {}
    for name, aa in aa_by_name.items():
        groups.setdefault(aa[lo:hi], []).append(name)
    n = len(aa_by_name)
    n_unique = sum(1 for m in groups.values() if len(m) == 1)
    return {
        "lo": lo, "hi": hi, "n_alleles": n,
        "n_classes": len(groups),
        "frac_uniquely_identifiable": (n_unique / n) if n else 0.0,
    }


def prefix_identifiability_curve(aa_by_name: dict, lengths: list) -> list:
    """Identifiability seeing only the first ``k`` residues, for each ``k`` in ``lengths``."""
    return [window_identifiability(aa_by_name, 0, k) for k in lengths]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_audit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/audit.py tests/alignair/aa/test_audit.py
git commit -m "feat(aa): identifiability + coverage-conditioned audit primitives"
```

---

### Task 6: Frozen-report assembler (`aa/report.py`)

Pure functions that assemble the Phase-0 metric dicts into one report object and render JSON + Markdown. No simulation here (that is Task 7) — keeps this unit-testable and deterministic.

**Files:**
- Create: `src/alignair/aa/report.py`
- Test: `tests/alignair/aa/test_report.py`

**Interfaces:**
- Produces:
  - `assemble_report(*, meta: dict, identifiability: dict, length_stats: dict, residue_stats: dict, boundary_stats: dict, d_stats: dict, coverage_curve: dict) -> dict` (top-level key `schema == "alignair.aa.phase0.v1"`).
  - `render_markdown(report: dict) -> str`.
  - `write_report(report: dict, out_dir: str) -> str` (writes `report.json` + `report.md`, returns `out_dir`).

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_report.py
import json
from alignair.aa.report import assemble_report, render_markdown, write_report


def _toy_report():
    return assemble_report(
        meta={"dc": "HUMAN_IGH_OGRDB", "n": 200, "seed": 0, "db_hash": "deadbeefdeadbeef"},
        identifiability={"V": {"n_alleles": 198, "n_classes": 150,
                               "frac_alleles_uniquely_identifiable": 0.6},
                         "J": {"n_alleles": 7, "n_classes": 6,
                               "frac_alleles_uniquely_identifiable": 0.71}},
        length_stats={"aa_len_p50": 120, "aa_len_p95": 128, "n_records": 190},
        residue_stats={"frac_X": 0.0, "frac_stop": 0.0},
        boundary_stats={"V_end_cut_frac": 0.66, "J_start_cut_frac": 0.66},
        d_stats={"mean_owned_d_residues": 1.4, "frac_zero_owned_d": 0.35},
        coverage_curve={"V": [{"lo": 0, "hi": 60, "frac_uniquely_identifiable": 0.3}]},
    )


def test_assemble_report_has_schema_and_sections():
    r = _toy_report()
    assert r["schema"] == "alignair.aa.phase0.v1"
    for key in ("meta", "identifiability", "length_stats",
                "residue_stats", "boundary_stats", "d_stats", "coverage_curve"):
        assert key in r


def test_render_markdown_mentions_key_numbers():
    md = render_markdown(_toy_report())
    assert "Phase 0" in md
    assert "HUMAN_IGH_OGRDB" in md
    assert "identifiab" in md.lower()


def test_write_report_emits_json_and_md(tmp_path):
    out = write_report(_toy_report(), str(tmp_path / "phase0"))
    with open(f"{out}/report.json") as f:
        loaded = json.load(f)
    assert loaded["schema"] == "alignair.aa.phase0.v1"
    assert (tmp_path / "phase0" / "report.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_report.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.aa.report'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/report.py
"""Assemble + render the frozen Phase-0 diagnostic report (no simulation here)."""
from __future__ import annotations

import json
import os


def assemble_report(*, meta, identifiability, length_stats, residue_stats,
                    boundary_stats, d_stats, coverage_curve) -> dict:
    return {
        "schema": "alignair.aa.phase0.v1",
        "meta": meta,
        "identifiability": identifiability,
        "length_stats": length_stats,
        "residue_stats": residue_stats,
        "boundary_stats": boundary_stats,
        "d_stats": d_stats,
        "coverage_curve": coverage_curve,
    }


def render_markdown(report: dict) -> str:
    m = report["meta"]
    idn = report["identifiability"]
    ls = report["length_stats"]
    rs = report["residue_stats"]
    bs = report["boundary_stats"]
    ds = report["d_stats"]
    lines = [
        "# Amino-Acid AlignAIR — Phase 0 Diagnostic Report",
        "",
        f"- Dataconfig: `{m.get('dc')}`  |  simulated records: {m.get('n')}  |  seed: {m.get('seed')}",
        f"- Germline DB hash: `{m.get('db_hash')}`",
        "",
        "## Identifiability (go/no-go gate)",
    ]
    for gene in ("V", "J"):
        g = idn.get(gene, {})
        lines.append(
            f"- {gene}: {g.get('n_classes')}/{g.get('n_alleles')} protein classes; "
            f"uniquely identifiable fraction = {g.get('frac_alleles_uniquely_identifiable')}"
        )
    lines += [
        "",
        "## Sequence & residue audit",
        f"- AA length p50/p95: {ls.get('aa_len_p50')}/{ls.get('aa_len_p95')} "
        f"over {ls.get('n_records')} records",
        f"- unknown-residue fraction: {rs.get('frac_X')}  |  stop fraction: {rs.get('frac_stop')}",
        "",
        "## Boundary & D-segment audit",
        f"- V-end boundary cuts a codon in {bs.get('V_end_cut_frac')} of reads; "
        f"J-start in {bs.get('J_start_cut_frac')}",
        f"- mean fully-owned D residues: {ds.get('mean_owned_d_residues')}; "
        f"reads with 0 owned D residues: {ds.get('frac_zero_owned_d')}",
    ]
    return "\n".join(lines) + "\n"


def write_report(report: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(render_markdown(report))
    return out_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_report.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/report.py tests/alignair/aa/test_report.py
git commit -m "feat(aa): Phase-0 report assembler + markdown renderer"
```

---

### Task 7: Simulation-driven Phase-0 builder + CLI (`aa/phase0.py`, `scripts/aa/phase0_report.py`)

Tie the primitives together: stream GenAIRR IGH records, project codon provenance, and compute every audit statistic, returning a report dict. A thin CLI writes the frozen artifact. This is the one place that runs simulation, so its test is `@pytest.mark.slow` with a small `n`.

**Files:**
- Create: `src/alignair/aa/phase0.py`
- Create: `scripts/aa/phase0_report.py`
- Test: `tests/alignair/aa/test_phase0.py`

**Interfaces:**
- Consumes: `translation.frame_offset_of`, `translation.project_codons`, `reference.build_aa_germlines`, `equivalence.db_hash`, `equivalence.build_equivalence_classes`, `audit.identifiability_summary`, `audit.prefix_identifiability_curve`, `report.assemble_report`.
- Produces:
  - `build_phase0_report(n: int, seed: int = 0, rate: float = 0.05, productive_only: bool = True, dc_name: str = "HUMAN_IGH_OGRDB") -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_phase0.py
import pytest


@pytest.mark.slow
def test_build_phase0_report_smoke_small_n():
    from alignair.aa.phase0 import build_phase0_report
    r = build_phase0_report(n=200, seed=0, rate=0.05, productive_only=True)
    assert r["schema"] == "alignair.aa.phase0.v1"
    # identifiability computed for V and J from real germlines
    assert r["identifiability"]["V"]["n_alleles"] == 198
    assert r["identifiability"]["J"]["n_alleles"] == 7
    assert 0.0 <= r["identifiability"]["V"]["frac_alleles_uniquely_identifiable"] <= 1.0
    # simulation-derived stats present and sane
    assert r["length_stats"]["n_records"] > 0
    assert 0.0 <= r["boundary_stats"]["V_end_cut_frac"] <= 1.0
    assert r["d_stats"]["mean_owned_d_residues"] >= 0.0
    assert r["meta"]["db_hash"] and len(r["meta"]["db_hash"]) == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_phase0.py -q -m slow`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.aa.phase0'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/phase0.py
"""Simulation-driven Phase-0 report builder: stream GenAIRR IGH records, project codon
provenance, and compute the identifiability / length / boundary / D audits."""
from __future__ import annotations

import statistics

from .translation import frame_offset_of, project_codons
from .reference import build_aa_germlines
from .equivalence import db_hash, build_equivalence_classes
from .audit import identifiability_summary, prefix_identifiability_curve
from .report import assemble_report

_SEG_COLS = {
    "V": ("v_sequence_start", "v_sequence_end"),
    "D": ("d_sequence_start", "d_sequence_end"),
    "J": ("j_sequence_start", "j_sequence_end"),
}


def _segments(rec) -> dict:
    segs = {}
    for seg, (a, b) in _SEG_COLS.items():
        s, e = rec.get(a), rec.get(b)
        if s is not None and e is not None:
            segs[seg] = (int(s), int(e))
    return segs


def build_phase0_report(n: int, seed: int = 0, rate: float = 0.05,
                        productive_only: bool = True,
                        dc_name: str = "HUMAN_IGH_OGRDB") -> dict:
    import GenAIRR as ga

    dc = getattr(ga, dc_name)

    # ---- germline-derived identifiability (no simulation needed) ----
    gset = build_aa_germlines(dc, genes=("V", "J"))
    dbh = db_hash(
        [(a.name, a.ungapped_seq) for a in dc.allele_list("v")]
        + [(a.name, a.ungapped_seq) for a in dc.allele_list("d")]
        + [(a.name, a.ungapped_seq) for a in dc.allele_list("j")]
    )
    identifiability = {}
    coverage_curve = {}
    for gene in ("V", "J"):
        aa_by_name = {n_: g.aa_seq for n_, g in gset.genes[gene].items()}
        cmap = build_equivalence_classes(gene, aa_by_name, dbh)
        identifiability[gene] = identifiability_summary(cmap)
        if gene == "V":
            lengths = [30, 60, 90, 120, max((len(s) for s in aa_by_name.values()), default=0)]
            coverage_curve[gene] = prefix_identifiability_curve(aa_by_name, lengths)

    # ---- simulation-derived stats ----
    exp = ga.Experiment.on(dc_name).recombine().mutate(model="s5f", rate=rate)
    aa_lengths = []
    n_x = n_stop = n_aa_chars = 0
    v_end_cuts = j_start_cuts = 0
    owned_d = []
    zero_d = 0
    used = 0
    for rec in exp.stream_records(n=n, seed=seed):
        js = rec.get("junction_start")
        aa = rec.get("sequence_aa")
        if js is None or not aa:
            continue
        if productive_only and not rec.get("productive"):
            continue
        used += 1
        aa_lengths.append(len(aa))
        n_x += aa.count("X")
        n_stop += aa.count("*")
        n_aa_chars += len(aa)
        fo = frame_offset_of(js)
        segs = _segments(rec)
        proj = project_codons(len(rec["sequence"]), fo, segs)
        if "V" in proj.phase and proj.phase["V"]["end"] != 0:
            v_end_cuts += 1
        if "J" in proj.phase and proj.phase["J"]["start"] != 0:
            j_start_cuts += 1
        db = proj.aa_bounds.get("D")
        owned = (db[1] - db[0]) if db else 0
        owned_d.append(owned)
        if owned == 0:
            zero_d += 1

    def _pct(vals, q):
        if not vals:
            return 0
        s = sorted(vals)
        return s[min(len(s) - 1, int(q * len(s)))]

    length_stats = {
        "n_records": used,
        "aa_len_p50": _pct(aa_lengths, 0.50),
        "aa_len_p95": _pct(aa_lengths, 0.95),
        "aa_len_max": max(aa_lengths) if aa_lengths else 0,
    }
    residue_stats = {
        "frac_X": (n_x / n_aa_chars) if n_aa_chars else 0.0,
        "frac_stop": (n_stop / n_aa_chars) if n_aa_chars else 0.0,
    }
    boundary_stats = {
        "V_end_cut_frac": (v_end_cuts / used) if used else 0.0,
        "J_start_cut_frac": (j_start_cuts / used) if used else 0.0,
    }
    d_stats = {
        "mean_owned_d_residues": (statistics.fmean(owned_d) if owned_d else 0.0),
        "frac_zero_owned_d": (zero_d / used) if used else 0.0,
    }

    return assemble_report(
        meta={"dc": dc_name, "n": n, "seed": seed, "rate": rate,
              "productive_only": productive_only, "db_hash": dbh,
              "quarantined_v": len(gset.quarantined["V"]),
              "quarantined_j": len(gset.quarantined["J"])},
        identifiability=identifiability,
        length_stats=length_stats,
        residue_stats=residue_stats,
        boundary_stats=boundary_stats,
        d_stats=d_stats,
        coverage_curve=coverage_curve,
    )
```

```python
# scripts/aa/phase0_report.py
"""CLI: build + freeze the amino-acid AlignAIR Phase-0 diagnostic report.

Run: PYTHONPATH=src .venv/bin/python scripts/aa/phase0_report.py --n 200000 --out experiments/aa/phase0
"""
import argparse

from alignair.aa.phase0 import build_phase0_report
from alignair.aa.report import write_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Amino-acid AlignAIR Phase-0 diagnostic report")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--dc", default="HUMAN_IGH_OGRDB")
    ap.add_argument("--all", action="store_true", help="include non-productive reads")
    ap.add_argument("--out", default="experiments/aa/phase0")
    args = ap.parse_args()

    report = build_phase0_report(
        n=args.n, seed=args.seed, rate=args.rate,
        productive_only=not args.all, dc_name=args.dc,
    )
    out = write_report(report, args.out)
    print(f"wrote {out}/report.json and {out}/report.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_phase0.py -q -m slow`
Expected: PASS (streams 200 records, builds a full report).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/phase0.py scripts/aa/phase0_report.py tests/alignair/aa/test_phase0.py
git commit -m "feat(aa): simulation-driven Phase-0 report builder + CLI"
```

---

### Task 8: Run the audit at scale, freeze the report, record the go/no-go gate

Not code — this produces the deliverable and the decision. Runs the CLI at scale, commits the frozen artifact, and appends the gate verdict to the design spec.

**Files:**
- Create (generated): `experiments/aa/phase0/report.json`, `experiments/aa/phase0/report.md`
- Modify: `docs/superpowers/specs/2026-07-18-aminoacid-alignair-design.md` (append the Phase-0 verdict)

- [ ] **Step 1: Run the full test suite for the subpackage (fast + slow)**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/ -q`
Expected: PASS (all Tasks 1-7 tests green, including slow).

- [ ] **Step 2: Confirm the nucleotide package is untouched (no regressions)**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair -q -m "not slow"`
Expected: PASS — the AA subpackage is additive; existing tests are unaffected.

- [ ] **Step 3: Generate the frozen report at scale**

Run: `PYTHONPATH=src .venv/bin/python scripts/aa/phase0_report.py --n 200000 --seed 0 --out experiments/aa/phase0`
Expected: prints `wrote experiments/aa/phase0/report.json and experiments/aa/phase0/report.md`.

- [ ] **Step 4: Read the report and evaluate the gate**

Run: `cat experiments/aa/phase0/report.md`

Evaluate against the §7 Phase-0A gate. Record the actual numbers (do not pre-judge them) for:
- V/J protein-class counts and uniquely-identifiable fractions (full-length and coverage-conditioned).
- unknown/stop residue frequency (justifies the `X`/`*` vocab decisions).
- V-end / J-start codon-cut fractions (justifies boundary masking in Phase 1 supervision).
- mean fully-owned D residues + fraction of reads with zero owned D residues (justifies the D-as-source-classification / abstention decision in Phase 1b).

**Gate rule (from the spec):** proceed to Phase 1 only if enough V/J alleles remain distinguishable as protein classes to make the model useful. If identifiability is too low, Phase 1 is a NO-GO and the finding itself is the deliverable.

- [ ] **Step 5: Append the verdict to the spec and commit the artifact**

Append a short "Phase 0 — Results & Gate Verdict" section to the design spec with the real numbers and the GO / NO-GO decision, then:

```bash
git add experiments/aa/phase0/report.json experiments/aa/phase0/report.md docs/superpowers/specs/2026-07-18-aminoacid-alignair-design.md
git commit -m "chore(aa): freeze Phase-0 diagnostic report + gate verdict"
```

---

## Self-Review

**Spec coverage (Phase 0A + 0B sections of the design spec):**
- 0A "build protein-equivalence classes" → Task 4. ✅
- 0A "measure unique V/J/D identifiability, full-length + coverage-conditioned" → Task 5 (`identifiability_summary`, `window_identifiability`, `prefix_identifiability_curve`) + Task 7 wires V/J; **D identifiability** is reported as source-recovery via the D-owned-residue stats (Task 7 `d_stats`), consistent with §5.3 that D has no protein-equivalence class. ✅ (D is intentionally NOT given equivalence classes.)
- 0A "audit sequence lengths + unknown/stop frequencies" → Task 7 `length_stats` / `residue_stats`. ✅
- 0A "quantify how often each segment boundary cuts a codon" → Task 1 `phase` + Task 7 `boundary_stats`. ✅
- 0A "measure fully-owned D residues after trimming" → Task 1 `aa_bounds["D"]` + Task 7 `d_stats`. ✅
- 0B "implement translation + codon-provenance projection" → Tasks 1. ✅
- 0B "validate V/J conserved anchors; apply anchor QC" → Task 3. ✅
- 0B "fix vocabulary/invalid-input policy" → Task 2. ✅
- 0B "verify GenAIRR streaming exposes sequence_aa" → exercised in Task 7 (records filtered on `sequence_aa`); the slow test asserts records are used. ✅
- 0B "define D supervision policy" → documented as a decision (D excluded from equivalence/translation; measured via owned-residue stats). The concrete D-head design is Phase 1b, out of scope here. ✅
- 0B "frozen diagnostic report" → Tasks 6-8. ✅
- Global "no neural training" → honored (no torch model anywhere; torch used only for the tokenizer batch tensor). ✅
- Global "DB-version-stamped class map" → Task 4 `db_hash` + Task 7 stamps it into `meta`. ✅

**Placeholder scan:** none — every code step contains complete, runnable code; every run step has an exact command + expected result. The "record the real numbers" language in Task 8 is intentional (the numbers are the experimental output, not a placeholder for code).

**Type consistency:** `AAProjection.aa_bounds` / `.phase` shapes are produced in Task 1 and consumed identically in Task 7. `EquivalenceClassMap` (Task 4) is consumed by `identifiability_summary` (Task 5) and Task 7. `assemble_report` keyword args (Task 6) exactly match the call in Task 7. `build_aa_germlines` returns `AAGermlineSet` with `.genes[gene][name].aa_seq` used consistently in Task 7. `db_hash` signature (Task 4) matches its call in Task 7. ✅

**Scope check:** single subsystem (the `aa/` audit prototype), one implementation plan, produces a working + testable deliverable (the frozen report + gate). Phases 1/1b/2/3 are explicitly out of scope and get their own plans.
