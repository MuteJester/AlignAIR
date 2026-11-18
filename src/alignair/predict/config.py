"""Configuration for the prediction pipeline."""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING

from ..core.boundaries import read_window
from ..core.tokens import DEFAULT_WINDOW
from ..errors import ConfigurationError
from ..reference.genes import model_genes

if TYPE_CHECKING:
    from ..core.model import AlignAIR
    from ..reference.reference_set import ReferenceSet


@dataclass
class PredictConfig:
    """How the prediction pipeline (``predict.predict``) runs a model on reads: the model's window and D presence, how
    many alleles a call keeps, the batch size, whether records get the full AIRR fields, and the reference's loci.
    ConfigurationError at construction for a threshold that is not a finite number, or a cap or batch size that is not
    a positive integer."""
    max_seq_length: int = DEFAULT_WINDOW
    has_d: bool = True
    threshold: float = 0.5            # the called set keeps the alleles scoring at least this (relative to the best)
    cap: int = 3                      # max alleles per gene
    batch_size: int = 64
    airr: bool = True                 # assemble full AIRR fields (junction/CDR/FWR/np/identity); off = light records
    loci: tuple | None = None      # the reference's ordered locus names; one locus labels every record

    def __post_init__(self) -> None:
        for field in ("cap", "batch_size"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ConfigurationError(f"{field} must be a positive integer (got {value!r})",
                                         context={"field": field})
        threshold = self.threshold
        if isinstance(threshold, bool) or not isinstance(threshold, Real) or not math.isfinite(threshold):
            raise ConfigurationError(f"threshold must be a finite number (got {self.threshold!r})",
                                     context={"field": "threshold"})

    @property
    def window(self) -> int:
        """The longest read the model sees: the padded width, less the reserved end slot."""
        return read_window(self.max_seq_length)

    @property
    def genes(self) -> tuple[str, ...]:
        """The genes the model calls: V, D and J, or V and J without a D."""
        return model_genes(self.has_d)

    @classmethod
    def for_model(cls, model: AlignAIR, reference: ReferenceSet, *, batch_size: int, **overrides) -> PredictConfig:
        """The configuration ``Aligner.predict`` predicts ``model`` with: its window and D presence,
        ``batch_size``, and the reference's ordered loci (None for a schema-less reference) unless ``overrides`` names
        them; an override that is None keeps the default."""
        overrides = {key: value for key, value in overrides.items() if value is not None}
        overrides.setdefault("loci", reference.locus_names() or None)
        return cls(max_seq_length=model.cfg.max_seq_length, has_d=model.cfg.has_d, batch_size=batch_size,
                   **overrides)
