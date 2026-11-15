"""Configuration for the prediction pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core.boundaries import read_window
from ..reference.genes import model_genes
from ..core.tokens import DEFAULT_WINDOW


@dataclass
class PredictConfig:
    max_seq_length: int = DEFAULT_WINDOW
    has_d: bool = True
    threshold: float = 0.5            # the called set keeps the alleles scoring at least this (relative to the best)
    cap: int = 3                      # max alleles per gene
    batch_size: int = 64
    airr: bool = True                 # assemble full AIRR fields (junction/CDR/FWR/np/identity); off = light records
    chain_types: Optional[tuple] = None  # ordered locus names of the reference; one locus labels every record

    @property
    def window(self) -> int:
        """The longest read the model sees: the padded width, less the reserved end slot."""
        return read_window(self.max_seq_length)

    @property
    def genes(self) -> tuple[str, ...]:
        """The genes the model calls: V, D and J, or V and J without a D."""
        return model_genes(self.has_d)

    @classmethod
    def for_model(cls, model, reference, *, batch_size: int, **overrides) -> "PredictConfig":
        """The configuration ``alignair.predict_sequences`` predicts ``model`` with: its window and D presence,
        ``batch_size``, and the reference's ordered loci as the chain types (None for a schema-less reference) unless
        ``overrides`` names them."""
        overrides.setdefault("chain_types", (reference.locus_names() or None) if hasattr(reference, "locus_names")
                             else None)
        return cls(max_seq_length=model.cfg.max_seq_length, has_d=model.cfg.has_d, batch_size=batch_size,
                   **overrides)
