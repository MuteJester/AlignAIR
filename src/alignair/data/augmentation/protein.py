"""AA-native corruption operators — the REAL protein-input error model (mass-spec / Edman /
engineered sequences), as opposed to nucleotide frameshift. Used to augment training and to build
evaluation strata: residue substitutions, single-residue indels, I/L ambiguity, and truncation
(partial peptide coverage). All are deterministic given a numpy Generator."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real

from ...errors import ConfigurationError


@dataclass(frozen=True)
class ProteinSubstitutionPolicy:
    """Training-only, length-preserving substitutions; rates are sampled once per selected read."""
    probability: float = .5
    min_rate: float = .02
    max_rate: float = .10

    def __post_init__(self):
        for name in ('probability', 'min_rate', 'max_rate'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ConfigurationError('substitution probabilities and rates must be finite numbers in [0, 1]',
                                         context={'field': name})
            object.__setattr__(self, name, float(value))
        if self.min_rate > self.max_rate:
            raise ConfigurationError('minimum substitution rate must not exceed maximum rate')

    def apply(self, sequence: str, rng) -> str:
        if self.probability and rng.random() < self.probability:
            return aa_substitute(sequence, float(rng.uniform(self.min_rate, self.max_rate)), rng)
        return sequence

    def to_dict(self) -> dict:
        return {'policy': 'protein-substitution-v1', **asdict(self)}


_AA = "ACDEFGHIKLMNPQRSTVWY"


def aa_substitute(seq: str, rate: float, rng) -> str:
    """Replace each canonical residue with a random residue with probability ``rate``."""
    if rate <= 0:
        return seq
    s = list(seq)
    for i, ch in enumerate(s):
        if ch in _AA and rng.random() < rate:
            s[i] = _AA[int(rng.integers(20))]
    return "".join(s)


def aa_indel(seq: str, rng, n: int = 1) -> str:
    """Apply ``n`` single-residue insertions/deletions (each 50/50) at random positions."""
    s = seq
    for _ in range(n):
        if not s:
            break
        i = int(rng.integers(len(s)))
        if rng.random() < 0.5 and len(s) > 1:
            s = s[:i] + s[i + 1:]                       # deletion
        else:
            s = s[:i] + _AA[int(rng.integers(20))] + s[i:]   # insertion
    return s


def il_ambiguity(seq: str, symbol: str = "I") -> str:
    """Collapse isoleucine (I) and leucine (L) to one symbol — mass spectrometry cannot tell them
    apart (identical mass; IUPAC code J). We default to ``I`` (a real residue) so the collapsed
    sequence stays valid for both the AA tokenizer and the BLOSUM aligner."""
    return "".join(symbol if c in "IL" else c for c in seq)


def aa_truncate(seq: str, min_frac: float, max_frac: float, rng) -> str:
    """Keep a random contiguous window of length in ``[min_frac, max_frac] * len`` (partial coverage)."""
    n = len(seq)
    if n == 0:
        return seq
    lo = max(1, int(round(min_frac * n)))
    hi = max(lo, int(round(max_frac * n)))
    keep = int(rng.integers(lo, hi + 1))
    start = int(rng.integers(0, n - keep + 1))
    return seq[start:start + keep]
